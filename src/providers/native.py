"""Optional llama.cpp native endpoints.

The core application only requires OpenAI-compatible ``/v1/*`` routes.  When a
local server also exposes llama.cpp native routes, this client uses them for
exact token counts, utility generation, prompt inspection, FIM, status, and
LoRA management.  Every method returns ``None`` (or a neutral value) instead of
raising, so callers can fall back without failing a turn.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from .endpoint_telemetry import record_endpoint_request

HEALTH_TIMEOUT_SECONDS = 0.75
DEFAULT_TIMEOUT_SECONDS = 5.0
UNSUPPORTED_STATUS = frozenset({404, 405, 501})
METRIC_KEYS = frozenset(
    {
        "llamacpp:tokens_predicted_total",
        "llamacpp:prompt_tokens_total",
        "llamacpp:requests_processing",
        "llamacpp:kv_cache_usage_ratio",
        "llamacpp:kv_cache_tokens",
    }
)


@dataclass
class NativeProfile:
    """What a server said about itself."""

    base_url: str
    detected: bool = False
    ready: bool = False
    loading: bool = False
    props: dict[str, Any] = field(default_factory=dict)
    unsupported: set[str] = field(default_factory=set)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class NativeClient:
    """Bounded, failure-tolerant client for llama.cpp root routes."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        health_timeout: float = HEALTH_TIMEOUT_SECONDS,
        headers: Mapping[str, str] | None = None,
        client_factory: Callable[..., Any] | None = None,
        enabled: bool = True,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.health_timeout = health_timeout
        self.headers = dict(headers or {})
        self.client_factory = client_factory or httpx.Client
        self.enabled = enabled
        self.profile = NativeProfile(base_url=self.base_url)

    @classmethod
    def from_endpoint(cls, endpoint: Any, **kwargs: Any) -> NativeClient:
        return cls(
            str(getattr(endpoint, "base_url", "")),
            headers=getattr(endpoint, "headers", lambda: {})(),
            **kwargs,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        timeout: float | None = None,
        allow_unsupported: bool = False,
    ) -> Any | None:
        if not self.enabled:
            return None
        operation = path.strip("/").split("/", 1)[0]
        if not allow_unsupported and operation in self.profile.unsupported:
            return None
        client = None
        started = time.perf_counter()
        try:
            client = self.client_factory(
                headers=self.headers,
                timeout=timeout or self.timeout,
            )
            response = client.request(method, self._url(path), json=json)
            if response.status_code in UNSUPPORTED_STATUS:
                self.profile.unsupported.add(operation)
                record_endpoint_request(
                    operation,
                    self.base_url,
                    success=False,
                    elapsed_seconds=time.perf_counter() - started,
                    error_type="unsupported",
                )
                return None
            response.raise_for_status()
            record_endpoint_request(
                operation,
                self.base_url,
                success=True,
                elapsed_seconds=time.perf_counter() - started,
            )
            return response
        except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
            record_endpoint_request(
                operation,
                self.base_url,
                success=False,
                elapsed_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
            )
            return None
        finally:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except (httpx.HTTPError, OSError):
                    pass

    def health(self) -> str:
        """Return ``ready``, ``loading``, or ``absent``."""
        if not self.enabled:
            return "absent"
        client = None
        started = time.perf_counter()
        try:
            client = self.client_factory(
                headers=self.headers,
                timeout=self.health_timeout,
            )
            response = client.get(self._url("health"))
            record_endpoint_request(
                "health",
                self.base_url,
                success=response.status_code == 200,
                elapsed_seconds=time.perf_counter() - started,
                error_type=None if response.status_code == 200 else "not_ready",
            )
            if response.status_code == 200:
                self.profile.detected = True
                self.profile.ready = True
                self.profile.loading = False
                return "ready"
            if response.status_code == 503:
                self.profile.detected = True
                self.profile.ready = False
                self.profile.loading = True
                return "loading"
            if response.status_code in UNSUPPORTED_STATUS:
                self.profile.unsupported.add("health")
            return "absent"
        except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
            record_endpoint_request(
                "health",
                self.base_url,
                success=False,
                elapsed_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
            )
            return "absent"
        finally:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except (httpx.HTTPError, OSError):
                    pass

    def props(self) -> dict[str, Any] | None:
        response = self._request("GET", "props")
        if response is None:
            return None
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        if not isinstance(body, dict):
            return None
        self.profile.props = body
        self.profile.detected = True
        return body

    @staticmethod
    def _token_list(body: Any) -> list[int] | None:
        if not isinstance(body, dict):
            return None
        tokens = body.get("tokens")
        if not isinstance(tokens, list):
            return None
        result = []
        for item in tokens:
            if isinstance(item, bool):
                return None
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, list) and item and isinstance(item[0], int):
                result.append(item[0])
            else:
                return None
        return result

    def tokenize(
        self,
        content: str | list[str],
        *,
        add_special: bool = False,
        parse_special: bool = False,
        with_pieces: bool = False,
    ) -> list[int] | None:
        response = self._request(
            "POST",
            "tokenize",
            json={
                "content": content,
                "add_special": add_special,
                "parse_special": parse_special,
                "with_pieces": with_pieces,
            },
        )
        if response is None:
            return None
        try:
            return self._token_list(response.json())
        except (ValueError, TypeError):
            return None

    def detokenize(self, tokens: Iterable[int]) -> str | None:
        values = [int(token) for token in tokens]
        if not values:
            return ""
        response = self._request("POST", "detokenize", json={"tokens": values})
        if response is None:
            return None
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        if isinstance(body, dict) and isinstance(body.get("content"), str):
            return body["content"]
        return None

    def apply_template(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        payload: dict[str, Any] = {}
        if messages is not None:
            payload["messages"] = messages
        if prompt is not None:
            payload["prompt"] = prompt
        if tools is not None:
            payload["tools"] = tools
        response = self._request("POST", "apply-template", json=payload)
        if response is None:
            return None
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        if isinstance(body, dict) and isinstance(body.get("prompt"), str):
            return body["prompt"]
        if isinstance(body, str):
            return body
        return None

    def infill(
        self,
        input_prefix: str,
        input_suffix: str,
        *,
        n_predict: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        id_slot: int | None = None,
        cache_prompt: bool = True,
        stream: bool = False,
    ) -> str | None:
        payload = {
            "input_prefix": input_prefix,
            "input_suffix": input_suffix,
            "n_predict": max(1, int(n_predict)),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "cache_prompt": bool(cache_prompt),
            "stream": bool(stream),
        }
        if stop:
            payload["stop"] = list(stop)
        if id_slot is not None:
            payload["id_slot"] = int(id_slot)
        response = self._request("POST", "infill", json=payload)
        if response is None:
            return None
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        if isinstance(body, dict) and isinstance(body.get("content"), str):
            return body["content"]
        return None

    def slots(self) -> list[dict[str, Any]] | None:
        response = self._request("GET", "slots")
        if response is None:
            return None
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        if isinstance(body, list):
            return [slot for slot in body if isinstance(slot, dict)]
        if isinstance(body, dict) and isinstance(body.get("slots"), list):
            return [slot for slot in body["slots"] if isinstance(slot, dict)]
        return None

    def metrics(self) -> dict[str, float]:
        """Parse only whitelisted Prometheus counters; missing metrics are empty."""
        response = self._request("GET", "metrics")
        if response is None:
            return {}
        text = getattr(response, "text", "")
        result: dict[str, float] = {}
        for line in str(text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip(), _number(parts[1])
            if key in METRIC_KEYS and value is not None:
                result[key] = value
        return result

    @staticmethod
    def _adapters(body: Any) -> list[dict[str, Any]] | None:
        if isinstance(body, dict):
            body = body.get("adapters")
        if not isinstance(body, list):
            return None
        result = []
        for item in body:
            if not isinstance(item, dict):
                continue
            adapter_id = item.get("id")
            if isinstance(adapter_id, bool) or not isinstance(adapter_id, int):
                continue
            result.append(
                {
                    "id": adapter_id,
                    "path": item.get("path") if isinstance(item.get("path"), str) else "",
                    "scale": _number(item.get("scale")),
                }
            )
        return result

    def lora_adapters(self) -> list[dict[str, Any]] | None:
        response = self._request("GET", "lora-adapters")
        if response is None:
            return None
        try:
            return self._adapters(response.json())
        except (ValueError, TypeError):
            return None

    def set_lora(self, adapter_id: int, scale: float) -> bool:
        """Mutate a shared server scale only after the caller confirms."""
        response = self._request(
            "POST",
            "lora-adapters",
            json={"id": int(adapter_id), "scale": float(scale)},
        )
        return response is not None


def detect_native(
    base_url: str,
    *,
    client_factory: Callable[..., Any] | None = None,
    headers: Mapping[str, str] | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] | None = None,
) -> NativeProfile:
    """Detect native routes with bounded loading backoff."""
    import time

    sleep = sleep or time.sleep
    client = NativeClient(
        base_url,
        client_factory=client_factory,
        headers=headers,
    )
    state = client.health()
    attempts = 0
    while state == "loading" and attempts < max(0, retries):
        sleep(min(2.0, 0.25 * (2**attempts)))
        attempts += 1
        state = client.health()
    if state == "ready":
        client.props()
    return client.profile


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "HEALTH_TIMEOUT_SECONDS",
    "METRIC_KEYS",
    "NativeClient",
    "NativeProfile",
    "UNSUPPORTED_STATUS",
]
