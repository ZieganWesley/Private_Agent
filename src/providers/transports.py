"""Optional OpenAI-compatible completion and responses transports.

``/v1/chat/completions`` remains the default.  These clients are used only when
a caller explicitly selects them, so an endpoint that lacks one of the optional
routes is never called speculatively.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .endpoint_telemetry import record_endpoint_request
from .responses_protocol import (
    ResponsesStreamState,
    iter_sse_events,
    reply_from_body,
    tool_definitions,
)

DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass
class TransportResult:
    text: str
    model: str | None = None
    usage: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(api_key: str | None, headers: Mapping[str, str] | None) -> dict[str, str]:
    merged = {"Content-Type": "application/json", **dict(headers or {})}
    if api_key and "Authorization" not in merged:
        merged["Authorization"] = f"Bearer {api_key}"
    return merged


class CompletionsClient:
    """Call ``POST /v1/completions`` for utility generation."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: Callable[..., Any] | None = None,
        enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.client_factory = client_factory or httpx.Client
        self.enabled = enabled

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> TransportResult | None:
        if not self.enabled:
            return None
        started = time.perf_counter()

        def _record(ok: bool, error: str | None = None, usage=None) -> None:
            record_endpoint_request(
                "completions",
                self.base_url,
                success=ok,
                elapsed_seconds=time.perf_counter() - started,
                error_type=error,
                usage=usage,
            )

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
        }
        if stop:
            payload["stop"] = list(stop)
        client = None
        try:
            client = self.client_factory(
                headers=_headers(self.api_key, self.headers),
                timeout=self.timeout,
            )
            response = client.post(
                _endpoint(self.base_url, "completions"), json=payload
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or not choices:
                _record(False, "invalid_response")
                return None
            first = choices[0]
            text = first.get("text") if isinstance(first, dict) else None
            if not isinstance(text, str):
                _record(False, "invalid_response")
                return None
            usage = body.get("usage") if isinstance(body, dict) else None
            model_name = body.get("model") if isinstance(body, dict) else None
            _record(True, usage=usage if isinstance(usage, dict) else None)
            return TransportResult(
                text=text,
                model=model_name if isinstance(model_name, str) else model,
                usage=usage if isinstance(usage, dict) else None,
            )
        except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
            _record(False, type(exc).__name__)
            return None
        finally:
            if client is not None and hasattr(client, "close"):
                with contextlib.suppress(httpx.HTTPError, OSError):
                    client.close()


class ResponsesClient:
    """Call ``POST /v1/responses`` when the caller explicitly selects it."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: Callable[..., Any] | None = None,
        enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.client_factory = client_factory or httpx.Client
        self.enabled = enabled

    @staticmethod
    def _text(body: Any) -> str | None:
        if not isinstance(body, dict):
            return None
        output_text = body.get("output_text")
        if isinstance(output_text, str):
            return output_text
        output = body.get("output")
        if not isinstance(output, list):
            return None
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts) if parts else None

    def respond(
        self,
        input_text: str,
        *,
        model: str,
        instructions: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        tools: Any = None,
    ) -> TransportResult | None:
        if not self.enabled:
            return None
        started = time.perf_counter()

        def _record(ok: bool, error: str | None = None, usage=None) -> None:
            record_endpoint_request(
                "responses",
                self.base_url,
                success=ok,
                elapsed_seconds=time.perf_counter() - started,
                error_type=error,
                usage=usage,
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "max_output_tokens": max(1, int(max_output_tokens)),
            "temperature": float(temperature),
        }
        if instructions:
            payload["instructions"] = instructions
        definitions = tool_definitions(tools)
        if definitions:
            payload["tools"] = definitions
        client = None
        try:
            client = self.client_factory(
                headers=_headers(self.api_key, self.headers),
                timeout=self.timeout,
            )
            response = client.post(_endpoint(self.base_url, "responses"), json=payload)
            response.raise_for_status()
            body = response.json()
            parsed = reply_from_body(body)
            if parsed is None:
                _record(False, "invalid_response")
                return None
            usage = parsed["usage"]
            _record(True, usage=usage)
            return TransportResult(
                text=parsed["text"],
                model=parsed["model"] or model,
                usage=usage,
                tool_calls=parsed["tool_calls"],
            )
        except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
            _record(False, type(exc).__name__)
            return None
        finally:
            if client is not None and hasattr(client, "close"):
                with contextlib.suppress(httpx.HTTPError, OSError):
                    client.close()

    def stream_respond(
        self,
        input_text: str,
        *,
        model: str,
        instructions: str | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        tools: Any = None,
    ) -> Iterator[dict[str, Any]]:
        """Consume the Responses SSE stream and yield normalized deltas."""
        if not self.enabled:
            return
        started = time.perf_counter()

        def _record(ok: bool, error: str | None = None, usage=None) -> None:
            record_endpoint_request(
                "responses_stream",
                self.base_url,
                success=ok,
                elapsed_seconds=time.perf_counter() - started,
                error_type=error,
                usage=usage,
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "max_output_tokens": max(1, int(max_output_tokens)),
            "temperature": float(temperature),
            "stream": True,
        }
        if instructions:
            payload["instructions"] = instructions
        definitions = tool_definitions(tools)
        if definitions:
            payload["tools"] = definitions
        client = None
        try:
            client = self.client_factory(
                headers=_headers(self.api_key, self.headers),
                timeout=self.timeout,
            )
            state = ResponsesStreamState()
            with client.stream(
                "POST", _endpoint(self.base_url, "responses"), json=payload
            ) as response:
                response.raise_for_status()
                for event in iter_sse_events(response.iter_lines()):
                    for delta in state.handle(event):
                        if delta["type"] == "error":
                            raise RuntimeError(delta["error"])
                        if delta["type"] != "completed":
                            yield delta
            result = state.result()
            if result is None:
                raise RuntimeError("empty responses stream")
            usage = result["usage"]
            _record(True, usage=usage)
            yield {
                "type": "final",
                "result": TransportResult(
                    text=result["text"],
                    model=result["model"] or model,
                    usage=usage,
                    tool_calls=result["tool_calls"],
                ),
            }
        except (httpx.HTTPError, ValueError, TypeError, OSError, RuntimeError) as exc:
            _record(False, type(exc).__name__)
            raise RuntimeError(
                f"The responses stream failed ({type(exc).__name__})."
            ) from exc
        finally:
            if client is not None and hasattr(client, "close"):
                with contextlib.suppress(httpx.HTTPError, OSError):
                    client.close()


__all__ = [
    "CompletionsClient",
    "DEFAULT_TIMEOUT_SECONDS",
    "ResponsesClient",
    "TransportResult",
]
