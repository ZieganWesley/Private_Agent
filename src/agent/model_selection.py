# ruff: noqa: F821
"""Chat-model construction, capability inspection and online model picking."""

from __future__ import annotations

try:
    import ollama
except Exception:  # noqa: BLE001 - optional dependency
    ollama = None

import contextlib
import sys
from collections.abc import Sequence
from getpass import getpass
from typing import Any

from config import (
    APP_CONFIG,
    HARDWARE_ACCELERATION_MODE,
    INTERNET_CHECK_TIMEOUT,
    MAX_CONTEXT_TOKENS,
    MAX_MODEL_CAPABILITY_CACHE_ENTRIES,
    MAX_OUTPUT_TOKENS,
    MAX_TOOL_CALLS,
    MAX_TOOL_ITERATIONS,
    MODEL_TEMPERATURE,
    OLLAMA_BASE_URL,
    ONLINE_MAX_RETRIES,
    ONLINE_MODEL_LIST_LIMIT,
    ONLINE_MODEL_LIST_TIMEOUT,
    ONLINE_PERMISSION_OVERRIDE,
    ONLINE_REQUEST_TIMEOUT,
    PROVIDER_API_MODE,
)
from hardware import ollama_acceleration_options
from providers.endpoints import EndpointRegistry, is_loopback_url, remote_allowed

from .console import console
from .local_runtimes import LocalRuntime, _is_loopback_url, discover_local_runtimes
from .model_capabilities import detect_openai_compatible_capabilities
from .providers import (
    create_openai_compatible_chat_model as _create_openai_compatible_chat_model,
)
from .providers import create_robust_local_chat_model as _create_robust_local_chat_model
from .providers import select_online_model as _select_online_model
from .providers import validate_online_base_url as _validate_online_base_url
from providers.transport_model import TransportChatModel
from .session_state import SESSION_STATS as _SESSION_STATS

_MODEL_CAPABILITIES_CACHE: dict[tuple[str, ...], dict[str, bool | None | int]] = {}
_MODEL_CONSTRUCTION_COUNT: int = 0


def model_construction_count() -> int:
    """Return how many chat-model objects this process has constructed."""
    return _MODEL_CONSTRUCTION_COUNT


def _ollama_client(*args, **kwargs):
    """Lazily import and build an Ollama client (keeps ollama optional)."""
    import ollama as _ollama

    return _ollama.Client(*args, **kwargs)


def ChatOllama(*args, **kwargs):
    """Lazily import ChatOllama (keeps langchain_ollama optional)."""
    from langchain_ollama import ChatOllama as _ChatOllama

    return _ChatOllama(*args, **kwargs)


def _local_endpoint_capability_overrides(base_url: str) -> dict | None:
    """Return the configured `capabilities` override for a local endpoint."""
    model_config = APP_CONFIG.get("models", {})
    endpoints = []
    if isinstance(model_config, dict):
        endpoints.extend(model_config.get("local_openai_compatible_endpoints", []))
        endpoints.extend(model_config.get("endpoints", []))
    for entry in endpoints if isinstance(endpoints, list) else []:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("base_url"), str)
            and entry["base_url"].rstrip("/") == base_url.rstrip("/")
            and isinstance(entry.get("capabilities"), dict)
        ):
            return entry["capabilities"]
    return None


def inspect_openai_compatible_model_capabilities(
    runtime: LocalRuntime, model_name: str
) -> dict[str, bool | None | int]:
    """Detect capabilities of a model served by a local OpenAI-compatible runtime."""
    return detect_openai_compatible_capabilities(
        model_name,
        base_url=runtime.base_url,
        client_factory=public_only_sync_client,
        capability_cache=_MODEL_CAPABILITIES_CACHE,
        max_cache_entries=MAX_MODEL_CAPABILITY_CACHE_ENTRIES,
        overrides=_local_endpoint_capability_overrides(runtime.base_url),
    )


def _effective_context_window(capabilities: dict) -> int:
    declared = capabilities.get("context_window")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return min(MAX_CONTEXT_TOKENS, declared)
    return MAX_CONTEXT_TOKENS


def get_robust_chat_model(
    primary_model_name: str,
    fallback_model_name: str,
    tools: Sequence[Any] | None = None,
    *,
    temperature: float = MODEL_TEMPERATURE,
    base_url: str = OLLAMA_BASE_URL,
    thinking: bool | str | None = None,
):
    """Compatibility entry point for robust local model construction."""
    return _create_robust_local_chat_model(
        primary_model_name,
        fallback_model_name,
        tools,
        temperature=temperature,
        base_url=base_url,
        acceleration_mode=HARDWARE_ACCELERATION_MODE,
        thinking=thinking,
        chat_model_factory=ChatOllama,
        client_kwargs=ollama_langchain_client_kwargs,
        acceleration_options=ollama_acceleration_options,
        track_clients=track_ollama_http_clients,
        console=console,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        context_window=MAX_CONTEXT_TOKENS,
    )


def _make_online_chat_model(base_url: str, model_name: str, api_key: str):
    global _MODEL_CONSTRUCTION_COUNT
    _MODEL_CONSTRUCTION_COUNT += 1
    with contextlib.suppress(Exception):
        _SESSION_STATS.model_constructions += 1
    from langchain_openai import ChatOpenAI

    return _create_openai_compatible_chat_model(
        base_url,
        model_name,
        api_key,
        temperature=MODEL_TEMPERATURE,
        request_timeout=ONLINE_REQUEST_TIMEOUT,
        max_retries=ONLINE_MAX_RETRIES,
        max_tokens=min(MAX_OUTPUT_TOKENS, max(1, MAX_CONTEXT_TOKENS // 3)),
        chat_model_factory=ChatOpenAI,
        sync_client_factory=public_only_sync_client,
        async_client_factory=_public_only_async_client,
    )


def _make_local_openai_compatible_chat_model(base_url: str, model_name: str):
    """Build a local OpenAI-compatible client without a real credential."""
    return _make_online_chat_model(base_url, model_name, "local-runtime")


def _make_selected_local_chat_model(
    runtime: LocalRuntime,
    model_name: str,
    *,
    thinking_enabled: bool,
    thinking_effort: str,
    capabilities: dict,
    api_mode: str | None = None,
):
    """Build a chat model using the configured provider API surface."""
    mode = str(api_mode or PROVIDER_API_MODE).strip().lower()
    if mode in {"completions", "responses"}:
        return TransportChatModel(
            runtime.base_url,
            model_name,
            api_mode=mode,
            api_key="local-runtime",
            timeout=float(ONLINE_REQUEST_TIMEOUT),
            client_factory=public_only_sync_client,
        )
    if not _is_loopback_url(runtime.base_url):
        console.print(
            f"[warning][Remote host][/warning] {runtime.base_url} is not on this "
            "machine; prompts and model responses leave the local device."
        )
    return _make_local_openai_compatible_chat_model(runtime.base_url, model_name)


def _configured_endpoint_models(entry: dict[str, Any]) -> tuple[str, ...] | None:
    """Return explicitly configured model names for an endpoint, if any."""
    if not isinstance(entry, dict):
        return None
    raw_models = entry.get("models", entry.get("model"))
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    if isinstance(raw_models, list):
        models = tuple(
            str(model).strip()
            for model in raw_models
            if isinstance(model, str) and str(model).strip()
        )
        if models:
            return models
    return None


def _discover_available_local_runtimes() -> list[LocalRuntime]:
    """Probe OpenAI-compatible endpoints configured in the local config file."""
    registry = EndpointRegistry.from_config(APP_CONFIG)
    configured_entries = [
        {
            "name": endpoint.name,
            "base_url": endpoint.base_url,
            "models": list(endpoint.models),
        }
        for endpoint in registry.endpoints
        if endpoint.enabled and remote_allowed(endpoint)
    ]
    allow_remote = any(
        not is_loopback_url(endpoint.base_url)
        for endpoint in registry.endpoints
        if endpoint.enabled
    )
    runtimes = discover_local_runtimes(
        None,
        configured_entries,
        client_factory=public_only_sync_client,
        allow_remote=allow_remote,
    )

    # Apply explicitly configured model names, allowing provider-only entries to
    # use the models reported by the endpoint.
    by_url = {
        str(entry.get("base_url", "")).rstrip("/"): entry
        for entry in configured_entries
        if isinstance(entry.get("base_url"), str)
    }
    result: list[LocalRuntime] = []
    for runtime in runtimes:
        entry = by_url.get(runtime.base_url.rstrip("/"))
        configured_models = (
            _configured_endpoint_models(entry) if entry is not None else None
        )
        if configured_models is not None:
            result.append(
                LocalRuntime(
                    name=runtime.name,
                    kind="openai-compatible",
                    base_url=runtime.base_url,
                    models=configured_models,
                    embedding_models=runtime.embedding_models,
                )
            )
        else:
            result.append(
                LocalRuntime(
                    name=runtime.name,
                    kind="openai-compatible",
                    base_url=runtime.base_url,
                    models=runtime.models,
                    embedding_models=runtime.embedding_models,
                )
            )
    return result


def select_online_model(use_saved_details: bool | None = None):
    return _select_online_model(
        app_config=APP_CONFIG,
        console=console,
        is_interactive=sys.stdin.isatty,
        validate_base_url=_validate_online_base_url,
        check_connection=check_internet_connection,
        connection_timeout=INTERNET_CHECK_TIMEOUT,
        get_api_key=getpass,
        model_list_timeout=ONLINE_MODEL_LIST_TIMEOUT,
        model_list_limit=ONLINE_MODEL_LIST_LIMIT,
        public_sync_client=public_only_sync_client,
        model_factory=_make_online_chat_model,
        permission_override=ONLINE_PERMISSION_OVERRIDE,
        use_saved_details=use_saved_details,
        max_tool_iterations=MAX_TOOL_ITERATIONS,
        max_tool_calls=MAX_TOOL_CALLS,
    )


def fetch_local_chat_models() -> list[str]:
    """Return chat model names advertised by every reachable local runtime."""
    try:
        runtimes = _discover_available_local_runtimes()
    except (ValueError, OSError) as e:  # noqa: BLE001 - discovery boundary
        console.print(
            f"[error][Warning] Could not discover local runtimes: {e}[/error]"
        )
        return []
    chat_models: list[str] = []
    seen: set[str] = set()
    for runtime in runtimes:
        for model_name in getattr(runtime, "models", ()) or ():
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            chat_models.append(model_name)
    return chat_models


class _LazyProxy:
    __slots__ = ("_module", "_attr", "_value", "_loaded")

    def __init__(self, module_path, attr_name):
        self._module = module_path
        self._attr = attr_name
        self._value = None
        self._loaded = False

    def _load(self):
        if not self._loaded:
            import importlib

            module = importlib.import_module(self._module)
            self._value = getattr(module, self._attr)
            self._loaded = True
        return self._value

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._load(), name, value)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __bool__(self) -> bool:
        return bool(self._load())

    def __repr__(self):
        return repr(self._load())


for _name, _attr in {
    "_public_only_async_client": "_public_only_async_client",
    "check_internet_connection": "check_internet_connection",
    "ollama_client_kwargs": "ollama_client_kwargs",
    "ollama_langchain_client_kwargs": "ollama_langchain_client_kwargs",
    "public_only_sync_client": "public_only_sync_client",
    "track_ollama_http_clients": "track_ollama_http_clients",
}.items():
    globals()[_name] = _LazyProxy("tools", _attr)
