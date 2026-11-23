"""Bounded discovery for locally hosted language-model runtimes."""

import ipaddress
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from utils.run_logging import RUN_LOGGER, log_event

MAX_DISCOVERY_ENDPOINTS = 8
DISCOVERY_TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class LocalRuntime:
    """A reachable local model API and its advertised chat models."""

    name: str
    kind: str
    base_url: str
    models: tuple[str, ...]
    embedding_models: tuple[str, ...] = ()


def _is_loopback_url(url: str) -> bool:
    """Return whether a URL targets a loopback IP or localhost hostname."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return False
        host = parsed.hostname.rstrip(".").lower()
        if host == "localhost" or host.endswith(".localhost"):
            return True
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _extract_models(body: Any) -> tuple[str, ...]:
    """Read valid model identifiers from Ollama or OpenAI-compatible responses."""
    if not isinstance(body, dict):
        return ()
    entries = body.get("models", body.get("data", []))
    if not isinstance(entries, list):
        return ()
    identifiers = []
    for item in entries:
        if isinstance(item, dict):
            value = item.get("model") or item.get("name") or item.get("id")
            if isinstance(value, str) and value.strip():
                identifiers.append(value.strip())
    return tuple(dict.fromkeys(identifiers))


def _probe_endpoint(
    name: str,
    kind: str,
    base_url: str,
    path: str,
    client_factory: Callable[..., Any],
    timeout: float,
    *,
    allow_remote: bool = False,
) -> LocalRuntime | None:
    """Probe one loopback endpoint and return its local runtime when reachable."""
    if not allow_remote and not _is_loopback_url(base_url):
        log_event(
            RUN_LOGGER,
            "provider.local_endpoint_rejected",
            level=logging.WARNING,
            runtime_name=name,
            reason="non_loopback",
        )
        return None
    client = None
    try:
        client = client_factory(
            headers={},
            timeout=timeout,
            allow_loopback=True,
        )
        with client:
            response = client.get(f"{base_url.rstrip('/')}{path}")
            response.raise_for_status()
            all_models = _extract_models(response.json())
            embedding_markers = ("embed", "embedding", "bge", "e5", "nomic-embed")
            embedding_models = tuple(
                model
                for model in all_models
                if any(marker in model.lower() for marker in embedding_markers)
            )
            models = tuple(
                model
                for model in all_models
                if model not in embedding_models
            )
            return LocalRuntime(
                name, kind, base_url.rstrip("/"), models, embedding_models
            )
    except (httpx.HTTPError, ValueError, ConnectionError, OSError) as exc:
        log_event(
            RUN_LOGGER,
            "provider.local_discovery_failed",
            level=logging.DEBUG,
            runtime_name=name,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return None
    finally:
        if client is not None and not getattr(client, "is_closed", True):
            try:
                client.close()
            except (httpx.HTTPError, ValueError, OSError) as exc:
                log_event(
                    RUN_LOGGER,
                    "provider.local_discovery_client_close_failed",
                    level=logging.WARNING,
                    runtime_name=name,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )


def discover_local_runtimes(
    ollama_base_url: Any,
    openai_compatible_endpoints: Any,
    *,
    client_factory: Callable[..., Any],
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
    allow_remote: bool = False,
) -> list[LocalRuntime]:
    """Find reachable local model runtimes with bounded parallel probes.

    Args:
        ollama_base_url: Configured Ollama URL.
        openai_compatible_endpoints: Configured local OpenAI-compatible entries.
        client_factory: Policy-controlled HTTP client factory.
        timeout: Per-endpoint HTTP timeout in seconds.

    Returns:
        Reachable local runtimes, ordered by their configured endpoint order.

    Raises:
        ValueError: If timeout is not positive.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Discovery timeout must be positive.")

    ollama_urls = (
        ollama_base_url
        if isinstance(ollama_base_url, (list, tuple))
        else [ollama_base_url]
    )
    endpoints: list[tuple[str, str, str, str]] = []
    if isinstance(openai_compatible_endpoints, list):
        for entry in openai_compatible_endpoints[: MAX_DISCOVERY_ENDPOINTS - 1]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            base_url = entry.get("base_url")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(base_url, str) or not base_url.strip():
                continue
            endpoints.append(
                (
                    name.strip()[:64],
                    "openai-compatible",
                    base_url.rstrip("/"),
                    "/models",
                )
            )
    for index, url in enumerate(ollama_urls):
        if not isinstance(url, str) or not url.strip():
            continue
        name = (
            "Ollama" if index == 0 and len(ollama_urls) == 1 else f"Ollama {index + 1}"
        )
        endpoints.append((name, "ollama", url.rstrip("/"), "/api/tags"))

    unique_endpoints = []
    seen_urls = set()
    for endpoint in endpoints:
        if endpoint[2] in seen_urls:
            continue
        seen_urls.add(endpoint[2])
        unique_endpoints.append(endpoint)

    with ThreadPoolExecutor(
        max_workers=min(4, len(unique_endpoints)),
        thread_name_prefix="local-runtime-discovery",
    ) as executor:
        futures = [
            executor.submit(
                _probe_endpoint,
                name,
                kind,
                base_url,
                path,
                client_factory,
                timeout,
                allow_remote=allow_remote,
            )
            for name, kind, base_url, path in unique_endpoints
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except (httpx.HTTPError, ValueError, ConnectionError, OSError) as exc:
                log_event(
                    RUN_LOGGER,
                    "provider.local_discovery_task_failed",
                    level=logging.DEBUG,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )
    return [runtime for runtime in results if runtime is not None]
