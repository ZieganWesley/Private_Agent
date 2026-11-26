"""Endpoint registry for OpenAI-compatible model servers.

Every model request is addressed by URL.  An endpoint advertises one or more
roles (chat, embedding, rerank, native) so a session can use one server for
chat and another for embeddings without knowing which application is running.

The registry also converts the legacy ``models.local_openai_compatible_endpoints``
and ``models.ollama_base_url(s)`` keys into the same representation.  Legacy
keys are read once at startup and never used by request paths.
"""

from __future__ import annotations

import ipaddress
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

VALID_ROLES = ("chat", "embedding", "rerank", "native")
DEFAULT_ROLES = ("chat",)
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_PROBE_TTL_SECONDS = 300.0
MAX_NAME_CHARS = 64


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _clean_name(value: Any, fallback: str) -> str:
    name = str(value or "").strip()
    return (name or fallback)[:MAX_NAME_CHARS]


def normalize_base_url(value: Any) -> str | None:
    """Return a normalized HTTP(S) base URL, or ``None`` when invalid."""
    if not isinstance(value, str):
        return None
    raw = value.strip().rstrip("/")
    if not raw:
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    if parsed.path and not parsed.path.startswith("/"):
        return None
    return raw


def is_loopback_url(url: str) -> bool:
    """Return whether a URL points at localhost or a loopback address."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        host = parsed.hostname.rstrip(".").lower()
        if host == "localhost" or host.endswith(".localhost"):
            return True
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def with_v1(url: str) -> str:
    """Append ``/v1`` to a legacy server URL that has no API prefix."""
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


@dataclass(frozen=True)
class EndpointConfig:
    """One reachable OpenAI-compatible server and the roles it serves."""

    name: str
    base_url: str
    roles: tuple[str, ...] = DEFAULT_ROLES
    enabled: bool = True
    allow_remote: bool = False
    models: tuple[str, ...] = ()
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    request_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    auth_header_env: str | None = None
    tls_verify: bool = True
    source: str = "config"

    def url(self, path: str) -> str:
        """Join a path such as ``/chat/completions`` to the base URL."""
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url.rstrip('/')}{suffix}"

    def chat_url(self) -> str:
        return self.url("/chat/completions")

    def models_url(self) -> str:
        return self.url("/models")

    def headers(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return auth headers from an environment variable, never config."""
        if not self.auth_header_env:
            return {}
        env = os.environ if environ is None else environ
        value = env.get(self.auth_header_env, "")
        return {"Authorization": value} if value else {}

    def supports(self, role: str) -> bool:
        return role in self.roles


def _models(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []
    models = []
    for item in values:
        model = str(item).strip()
        if model and model not in models:
            models.append(model)
    return tuple(models)


def _roles(value: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []
    roles = []
    for item in values:
        role = str(item).strip().lower()
        if role in VALID_ROLES and role not in roles:
            roles.append(role)
    return tuple(roles) or fallback


def _endpoint_from_mapping(
    entry: Mapping[str, Any],
    *,
    index: int,
    source: str,
    default_roles: tuple[str, ...] = DEFAULT_ROLES,
    force_v1: bool = False,
) -> EndpointConfig | None:
    base_url = normalize_base_url(entry.get("base_url"))
    if base_url is None:
        return None
    if force_v1:
        base_url = with_v1(base_url)
    name = _clean_name(entry.get("name"), f"endpoint-{index + 1}")
    timeout = _positive_float(entry.get("request_timeout_seconds"))
    return EndpointConfig(
        name=name,
        base_url=base_url,
        roles=_roles(entry.get("roles"), default_roles),
        models=_models(entry.get("models", entry.get("model"))),
        enabled=_bool(entry.get("enabled"), True),
        allow_remote=_bool(entry.get("allow_remote"), False),
        max_context_tokens=_positive_int(entry.get("max_context_tokens")),
        max_output_tokens=_positive_int(entry.get("max_output_tokens")),
        request_timeout_seconds=timeout or DEFAULT_TIMEOUT_SECONDS,
        auth_header_env=(
            str(entry["auth_header_env"]).strip()
            if isinstance(entry.get("auth_header_env"), str)
            and entry["auth_header_env"].strip()
            else None
        ),
        tls_verify=_bool(entry.get("tls_verify"), True),
        source=source,
    )


@dataclass(frozen=True)
class EndpointRegistry:
    """An ordered, de-duplicated set of endpoints and migration warnings."""

    endpoints: tuple[EndpointConfig, ...] = ()
    warnings: tuple[str, ...] = ()
    probe_ttl_seconds: float = DEFAULT_PROBE_TTL_SECONDS

    def for_role(self, role: str) -> tuple[EndpointConfig, ...]:
        return tuple(
            endpoint
            for endpoint in self.endpoints
            if endpoint.enabled and endpoint.supports(role)
        )

    def resolve(
        self,
        role: str = "chat",
        *,
        preferred: str | None = None,
    ) -> EndpointConfig | None:
        """Resolve a role, preferring an endpoint name when one is configured."""
        candidates = self.for_role(role)
        wanted = str(preferred or "").strip().lower()
        if wanted:
            for endpoint in candidates:
                if endpoint.name.lower() == wanted:
                    return endpoint
        return candidates[0] if candidates else None

    def by_name(self, name: str) -> EndpointConfig | None:
        wanted = str(name or "").strip().lower()
        return next(
            (
                endpoint
                for endpoint in self.endpoints
                if endpoint.name.lower() == wanted
            ),
            None,
        )

    def for_base_url(self, base_url: str) -> EndpointConfig | None:
        normalized = normalize_base_url(base_url)
        return next(
            (
                endpoint
                for endpoint in self.endpoints
                if normalize_base_url(endpoint.base_url) == normalized
            ),
            None,
        )

    def all_urls(self) -> tuple[str, ...]:
        return tuple(endpoint.base_url for endpoint in self.endpoints)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> EndpointRegistry:
        """Build the registry from new keys, then legacy keys, once."""
        config = config if isinstance(config, Mapping) else {}
        models = config.get("models")
        models = models if isinstance(models, Mapping) else {}

        entries: list[EndpointConfig] = []
        warnings: list[str] = []

        raw_endpoints = models.get("endpoints")
        if isinstance(raw_endpoints, list):
            for index, entry in enumerate(raw_endpoints):
                if not isinstance(entry, Mapping):
                    warnings.append(
                        f"Ignoring models.endpoints[{index}]: expected a table."
                    )
                    continue
                endpoint = _endpoint_from_mapping(
                    entry, index=index, source="models.endpoints"
                )
                if endpoint is None:
                    warnings.append(
                        f"Ignoring models.endpoints[{index}]: invalid base_url."
                    )
                    continue
                entries.append(endpoint)
        elif raw_endpoints is not None:
            warnings.append("Ignoring models.endpoints: expected an array of tables.")

        legacy_endpoints = models.get("local_openai_compatible_endpoints")
        if isinstance(legacy_endpoints, list):
            for index, entry in enumerate(legacy_endpoints):
                if not isinstance(entry, Mapping):
                    warnings.append(
                        "Ignoring local_openai_compatible_endpoints entry: "
                        "expected a table."
                    )
                    continue
                endpoint = _endpoint_from_mapping(
                    entry,
                    index=index,
                    source="legacy.local_openai_compatible_endpoints",
                    default_roles=VALID_ROLES,
                )
                if endpoint is None:
                    warnings.append(
                        "Ignoring local_openai_compatible_endpoints entry: "
                        "invalid base_url."
                    )
                    continue
                entries.append(endpoint)
        elif legacy_endpoints is not None:
            warnings.append(
                "Ignoring local_openai_compatible_endpoints: expected an array."
            )

        raw_ollama = models.get("ollama_base_urls", models.get("ollama_base_url"))
        if isinstance(raw_ollama, str):
            raw_ollama = [raw_ollama]
        if isinstance(raw_ollama, list):
            for index, value in enumerate(raw_ollama):
                if not isinstance(value, str) or not value.strip():
                    continue
                endpoint = _endpoint_from_mapping(
                    {
                        "name": ("Ollama" if index == 0 else f"Ollama {index + 1}"),
                        "base_url": value,
                        "roles": list(VALID_ROLES),
                    },
                    index=index,
                    source="legacy.ollama_base_url",
                    default_roles=VALID_ROLES,
                    force_v1=True,
                )
                if endpoint is None:
                    warnings.append("Ignoring legacy ollama_base_url: invalid URL.")
                    continue
                entries.append(endpoint)

        deduped: list[EndpointConfig] = []
        seen: set[str] = set()
        for endpoint in entries:
            key = endpoint.base_url.rstrip("/").lower()
            if key in seen:
                warnings.append(
                    f"Ignoring duplicate endpoint {endpoint.name!r}: "
                    f"{endpoint.base_url}."
                )
                continue
            seen.add(key)
            deduped.append(endpoint)

        if not deduped and warnings:
            warnings.append("No usable OpenAI-compatible endpoint is configured.")

        ttl = _positive_float(models.get("endpoint_probe_ttl_seconds"))
        return cls(
            endpoints=tuple(deduped),
            warnings=tuple(warnings),
            probe_ttl_seconds=ttl or DEFAULT_PROBE_TTL_SECONDS,
        )


def clamp_limit(
    discovered: int | None,
    configured: int | None,
    ceiling: int,
) -> int:
    """Clamp a discovered/configured limit to a never-exceeded ceiling."""
    values = [
        value
        for value in (discovered, configured, ceiling)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    return min(values) if values else max(1, int(ceiling))


def resolved_limits(
    endpoint: EndpointConfig,
    *,
    ceiling_context: int,
    ceiling_output: int,
    discovered_context: int | None = None,
    discovered_output: int | None = None,
) -> tuple[int, int]:
    """Return the effective (context, output) limits for an endpoint."""
    context = clamp_limit(
        discovered_context, endpoint.max_context_tokens, ceiling_context
    )
    output = clamp_limit(discovered_output, endpoint.max_output_tokens, ceiling_output)
    return context, min(output, context)


def remote_allowed(endpoint: EndpointConfig) -> bool:
    """Return whether the endpoint may be used for a non-loopback URL."""
    return is_loopback_url(endpoint.base_url) or endpoint.allow_remote


__all__ = [
    "DEFAULT_PROBE_TTL_SECONDS",
    "DEFAULT_ROLES",
    "DEFAULT_TIMEOUT_SECONDS",
    "EndpointConfig",
    "EndpointRegistry",
    "VALID_ROLES",
    "clamp_limit",
    "is_loopback_url",
    "normalize_base_url",
    "remote_allowed",
    "resolved_limits",
    "with_v1",
]
