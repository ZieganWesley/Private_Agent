"""Chat-model construction, capability inspection and online model picking."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from getpass import getpass
from typing import Any

import ollama
from langchain_ollama import ChatOllama

from ..config import (
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
)
from ..hardware import ollama_acceleration_options
from ..run_logging import RUN_LOGGER, log_event
from ..tools import (
    _public_only_async_client,
    check_internet_connection,
    ollama_client_kwargs,
    ollama_langchain_client_kwargs,
    public_only_sync_client,
    track_ollama_http_clients,
)
from .console import console
from .local_runtimes import LocalRuntime, discover_local_runtimes
from .model_capabilities import detect_openai_compatible_capabilities
from .providers import create_local_chat_model as _create_local_chat_model
from .providers import (
    create_openai_compatible_chat_model as _create_openai_compatible_chat_model,
)
from .providers import create_robust_local_chat_model as _create_robust_local_chat_model
from .providers import (
    inspect_local_model_capabilities as _inspect_local_model_capabilities,
)
from .providers import select_online_model as _select_online_model
from .providers import validate_online_base_url as _validate_online_base_url

_MODEL_CAPABILITIES_CACHE: dict[tuple[str, ...], dict[str, bool | None | int]] = {}


def _close_ollama_client(client):
    try:
        client.close()
    except (ValueError, OSError) as exc:
        log_event(
            RUN_LOGGER,
            "provider.ollama_client_close_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        console.print(
            f"[yellow][Shutdown warning] Could not close Ollama client: "
            f"{type(exc).__name__}[/yellow]"
        )


def inspect_model_capabilities(
    model_name: str,
) -> dict[str, bool | None | int]:
    """Compatibility entry point for Ollama capability discovery."""
    return _inspect_local_model_capabilities(
        model_name,
        base_url=OLLAMA_BASE_URL,
        client_factory=ollama.Client,
        client_kwargs=ollama_client_kwargs,
        capability_cache=_MODEL_CAPABILITIES_CACHE,
        max_cache_entries=MAX_MODEL_CAPABILITY_CACHE_ENTRIES,
        console=console,
        close_client=_close_ollama_client,
    )


def _local_endpoint_capability_overrides(base_url: str) -> dict | None:
    """Return the configured `capabilities` override for a local endpoint."""
    model_config = APP_CONFIG.get("models", {})
    endpoints = (
        model_config.get("local_openai_compatible_endpoints", [])
        if isinstance(model_config, dict)
        else []
    )
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


def _make_chat_model(
    model_name: str,
    *,
    thinking_enabled: bool,
    thinking_effort: str,
    supports_thinking: bool | None,
    context_window: int | None = None,
):
    effective_context = min(MAX_CONTEXT_TOKENS, context_window or MAX_CONTEXT_TOKENS)
    return _create_local_chat_model(
        model_name,
        temperature=MODEL_TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
        acceleration_mode=HARDWARE_ACCELERATION_MODE,
        thinking=(thinking_effort if thinking_enabled else False)
        if supports_thinking
        else None,
        chat_model_factory=ChatOllama,
        client_kwargs=ollama_langchain_client_kwargs,
        acceleration_options=ollama_acceleration_options,
        track_clients=track_ollama_http_clients,
        max_output_tokens=min(MAX_OUTPUT_TOKENS, max(1, effective_context // 3)),
        context_window=effective_context,
    )


def _make_online_chat_model(base_url: str, model_name: str, api_key: str):
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
):
    """Build a chat model using the selected local runtime's supported API."""
    if runtime.kind == "ollama":
        return _make_chat_model(
            model_name,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
            supports_thinking=capabilities.get("thinking"),
            context_window=capabilities.get("context_window"),
        )
    return _make_local_openai_compatible_chat_model(runtime.base_url, model_name)


def _discover_available_local_runtimes() -> list[LocalRuntime]:
    """Probe configured loopback model servers using bounded HTTP requests."""
    model_config = APP_CONFIG.get("models", {})
    endpoints = (
        model_config.get("local_openai_compatible_endpoints", [])
        if isinstance(model_config, dict)
        else []
    )
    return discover_local_runtimes(
        OLLAMA_BASE_URL,
        endpoints,
        client_factory=public_only_sync_client,
    )


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
    client = None
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL, **ollama_client_kwargs())
        response = client.list()
        models_list = (
            response.get("models", [])
            if isinstance(response, dict)
            else getattr(response, "models", [])
        )
        chat_models = []
        embedding_keywords = ["embed", "embedding", "bge", "e5", "nomic-embed"]

        for m in models_list:
            model_name = (
                m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
            )
            if not model_name:
                continue
            if any(kw in model_name.lower() for kw in embedding_keywords):
                continue
            chat_models.append(model_name)
        return chat_models
    except (ValueError, OSError) as e:
        console.print(f"[red][Warning] Could not connect to Ollama: {e}[/red]")
        return []
    finally:
        if client is not None:
            _close_ollama_client(client)
