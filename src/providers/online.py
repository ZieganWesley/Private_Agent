"""Resolve configured online (remote) OpenAI-compatible endpoints.

Online providers are configured as a list, like local endpoints, so several
accounts or gateways can be described in one config file and one of them can be
selected by name.  A single ``online_base_url``/``online_api_key`` pair is still
accepted and is treated as the first endpoint.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_ONLINE_BASE_URL = "https://api.openai.com/v1"
MAX_NAME_CHARS = 64


@dataclass(frozen=True)
class OnlineEndpoint:
    """One remote OpenAI-compatible endpoint and its credentials."""

    name: str
    base_url: str
    api_key: str | None = None
    models: tuple[str, ...] = ()
    enabled: bool = True
    source: str = "config"

    def has_api_key(self) -> bool:
        return bool(self.api_key)


def normalize_online_base_url(value: Any) -> str | None:
    """Return a validated online base URL with a ``/v1`` suffix, or ``None``.

    Remote endpoints must use HTTPS; plain HTTP is accepted only for loopback
    hosts, which is what local gateways use during testing.
    """
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = urlsplit(raw)
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    host = parsed.hostname.lower()
    is_loopback = host in {"localhost", "127.0.0.1", "::1"} or host.endswith(
        ".localhost"
    )
    if parsed.scheme != "https" and not is_loopback:
        return None
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _clean_name(value: Any, fallback: str) -> str:
    name = str(value or "").strip()
    return (name or fallback)[:MAX_NAME_CHARS]


def _models(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
    return ()


def _api_key(entry: Mapping[str, Any], environ: Mapping[str, str]) -> str | None:
    direct = entry.get("api_key")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    env_name = entry.get("api_key_env")
    if isinstance(env_name, str) and env_name.strip():
        return (environ.get(env_name.strip()) or "").strip() or None
    return None


def _endpoint_from_mapping(
    entry: Mapping[str, Any],
    *,
    index: int,
    source: str,
    environ: Mapping[str, str],
) -> OnlineEndpoint | None:
    base_url = normalize_online_base_url(entry.get("base_url"))
    if base_url is None:
        return None
    enabled = entry.get("enabled", True)
    return OnlineEndpoint(
        name=_clean_name(entry.get("name"), f"online-{index + 1}"),
        base_url=base_url,
        api_key=_api_key(entry, environ),
        models=_models(entry.get("models", entry.get("model"))),
        enabled=enabled if isinstance(enabled, bool) else True,
        source=source,
    )


def _models_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    models = config.get("models")
    return models if isinstance(models, Mapping) else {}


def online_endpoints_from_config(
    config: Mapping[str, Any] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[OnlineEndpoint, ...]:
    """Return the deduplicated online endpoints described by ``config``.

    ``models.online_endpoints`` is read first; the legacy top-level
    ``online_base_url``/``online_api_key`` pair becomes the last endpoint when no
    entry already covers that URL.  When nothing is configured the public
    OpenAI-compatible default is returned so the interactive selection flow keeps
    its documented default.
    """
    config = config if isinstance(config, Mapping) else {}
    environ = os.environ if environ is None else environ
    models = _models_section(config)

    entries: list[OnlineEndpoint] = []
    raw_endpoints = models.get("online_endpoints", config.get("online_endpoints"))
    if isinstance(raw_endpoints, list):
        for index, entry in enumerate(raw_endpoints):
            if not isinstance(entry, Mapping):
                continue
            endpoint = _endpoint_from_mapping(
                entry,
                index=index,
                source="models.online_endpoints",
                environ=environ,
            )
            if endpoint is not None:
                entries.append(endpoint)

    legacy_url = models.get("online_base_url", config.get("online_base_url"))
    legacy_key = models.get("online_api_key", config.get("online_api_key"))
    legacy = _endpoint_from_mapping(
        {"name": "OpenAI", "base_url": legacy_url, "api_key": legacy_key},
        index=len(entries),
        source="legacy.online_base_url",
        environ=environ,
    )
    if legacy is not None and not any(
        endpoint.base_url.rstrip("/").lower() == legacy.base_url.rstrip("/").lower()
        for endpoint in entries
    ):
        entries.append(legacy)

    if not entries:
        entries.append(
            OnlineEndpoint(
                name="OpenAI",
                base_url=DEFAULT_ONLINE_BASE_URL,
                source="default",
            )
        )

    deduped: list[OnlineEndpoint] = []
    seen: set[str] = set()
    for endpoint in entries:
        key = endpoint.base_url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(endpoint)
    return tuple(deduped)


def resolve_online_endpoint(
    config: Mapping[str, Any] | None,
    *,
    preferred: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> OnlineEndpoint | None:
    """Return the enabled endpoint to use, preferring one by name.

    The name comes from ``preferred`` or from ``models.preferred_online_endpoint``.
    Endpoints without a configured key are still returned: a session may collect
    the key interactively.
    """
    models = _models_section(config if isinstance(config, Mapping) else {})
    wanted = str(
        preferred if preferred is not None else models.get("preferred_online_endpoint")
    ).strip()
    candidates = [
        endpoint
        for endpoint in online_endpoints_from_config(config, environ=environ)
        if endpoint.enabled
    ]
    if not candidates:
        return None
    if wanted:
        lowered = wanted.lower()
        for endpoint in candidates:
            if endpoint.name.lower() == lowered:
                return endpoint
    return candidates[0]


def has_configured_online_key(
    config: Mapping[str, Any] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether any online endpoint carries a non-empty API key."""
    return any(
        endpoint.has_api_key()
        for endpoint in online_endpoints_from_config(config, environ=environ)
    )


__all__ = [
    "DEFAULT_ONLINE_BASE_URL",
    "OnlineEndpoint",
    "has_configured_online_key",
    "normalize_online_base_url",
    "online_endpoints_from_config",
    "resolve_online_endpoint",
]
