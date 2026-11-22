"""Detect model capabilities from what OpenAI-compatible servers report.

No server reports everything, so each source fills in what it knows and the
results are merged in priority order. Anything no source reports stays None
(unknown) instead of being guessed.

Sources, highest priority first:

1. Config overrides for the endpoint.
2. llama.cpp ``GET /props``: chat template caps (tool calls, reasoning),
   ``modalities`` and the running context size ``n_ctx``.
3. LM Studio ``GET /api/v1/models`` or ``/api/v0/models``: model ``type``,
   ``capabilities`` and context lengths.
4. The model's entry in ``GET /v1/models``, which many servers extend:
   llama.cpp (``architecture``, ``meta.n_ctx_train``), vLLM
   (``max_model_len``) and OpenRouter-style APIs (``supported_parameters``,
   ``context_length``).

Server-specific routes (2 and 3) are only probed on loopback endpoints.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import quote

import httpx

from utils.run_logging import RUN_LOGGER, log_event

from .local_runtimes import DISCOVERY_TIMEOUT_SECONDS, _is_loopback_url

CAPABILITY_FLAGS = (
    "tools",
    "function_calls",
    "structured_output",
    "thinking",
    "vision",
    "audio",
)
CAPABILITY_KEYS = (*CAPABILITY_FLAGS, "context_window")

_TOOL_NAMES = frozenset({"tools", "tool_use", "tool_calling", "function_calling"})
_REASONING_NAMES = frozenset({"thinking", "reasoning"})
# Most specific runtime value first; training context is only a ceiling.
_CONTEXT_FIELDS = (
    "loaded_context_length",
    "context_length",
    "max_model_len",
    "max_context_length",
)
_LMSTUDIO_MODEL_PATHS = ("/api/v1/models", "/api/v0/models")

# Last-resort knowledge for model families whose servers report nothing.
# Names are matched case-insensitively as substrings, longest first, so a more
# specific entry always wins over a shorter one that also matches.
MODEL_FAMILY_CAPABILITIES: dict[str, dict[str, bool | int | None]] = {
    "qwen3": {"tools": True, "thinking": True, "vision": False, "audio": False},
    "qwen2.5-coder": {
        "tools": True,
        "thinking": False,
        "vision": False,
        "audio": False,
    },
    "qwen2.5-vl": {"tools": True, "thinking": False, "vision": True, "audio": False},
    "qwen2.5": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "llava": {"tools": False, "thinking": False, "vision": True, "audio": False},
    "llama3.2-vision": {
        "tools": True,
        "thinking": False,
        "vision": True,
        "audio": False,
    },
    "llama3": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "deepseek-r1": {"tools": False, "thinking": True, "vision": False, "audio": False},
    "deepseek-coder": {
        "tools": True,
        "thinking": False,
        "vision": False,
        "audio": False,
    },
    "mistral": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "mixtral": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "phi4": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "gemma": {"tools": False, "thinking": False, "vision": False, "audio": False},
    "granite": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "command-r": {"tools": True, "thinking": False, "vision": False, "audio": False},
    "nomic-embed": {"tools": False, "thinking": False, "vision": False, "audio": False},
}

_FAMILY_MATCH_ORDER = sorted(MODEL_FAMILY_CAPABILITIES, key=len, reverse=True)


def unknown_capabilities() -> dict[str, bool | int | None]:
    return dict.fromkeys(CAPABILITY_KEYS)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _lowered_names(values: Iterable[Any]) -> set[str]:
    return {str(value).lower() for value in values if isinstance(value, str)}


def merge_capabilities(
    *sources: dict | None,
) -> dict[str, bool | int | None]:
    """Merge capability dicts; for each key the first valid value wins.

    ``function_calls`` always mirrors ``tools``.
    """
    result = unknown_capabilities()
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in CAPABILITY_FLAGS:
            if result[key] is None and isinstance(source.get(key), bool):
                result[key] = source[key]
        if result["context_window"] is None:
            result["context_window"] = _positive_int(source.get("context_window"))
    result["function_calls"] = result["tools"]
    return result


def capabilities_from_family(model_name: str) -> dict[str, bool | int | None]:
    """Look up a known model family; unknown families stay None (unknown)."""
    result = unknown_capabilities()
    lowered = str(model_name or "").lower()
    if not lowered:
        return result
    for family in _FAMILY_MATCH_ORDER:
        if family in lowered:
            for key, value in MODEL_FAMILY_CAPABILITIES[family].items():
                result[key] = value
            break
    return result


def capabilities_from_model_entry(entry: Any) -> dict[str, bool | int | None]:
    """Read capabilities from one model entry of a model-listing response."""
    result = unknown_capabilities()
    if not isinstance(entry, dict):
        return result

    architecture = entry.get("architecture")
    if isinstance(architecture, dict) and isinstance(
        architecture.get("input_modalities"), list
    ):
        inputs = _lowered_names(architecture["input_modalities"])
        result["vision"] = "image" in inputs
        result["audio"] = "audio" in inputs

    # LM Studio separates vision models ("vlm") from text-only ones ("llm").
    if entry.get("type") == "vlm":
        result["vision"] = True
    elif entry.get("type") == "llm" and result["vision"] is None:
        result["vision"] = False

    # OpenRouter-style APIs list every request parameter the model accepts.
    if isinstance(entry.get("supported_parameters"), list):
        parameters = _lowered_names(entry["supported_parameters"])
        result["tools"] = "tools" in parameters
        result["structured_output"] = bool(
            parameters & {"structured_outputs", "response_format"}
        )
        result["thinking"] = bool(parameters & {"reasoning", "include_reasoning"})

    declared = entry.get("capabilities")
    if isinstance(declared, list):
        # A list names what the model has; a missing name is not proof of absence.
        names = _lowered_names(declared)
        if names & _TOOL_NAMES:
            result["tools"] = True
        if names & _REASONING_NAMES:
            result["thinking"] = True
        for flag in ("vision", "audio"):
            if flag in names:
                result[flag] = True
    elif isinstance(declared, dict):
        for name, value in declared.items():
            if not isinstance(value, bool):
                continue
            name = str(name).lower()
            if name in _TOOL_NAMES or name == "trained_for_tool_use":
                result["tools"] = value
            elif name in _REASONING_NAMES:
                result["thinking"] = value
            elif name in {"vision", "audio"}:
                result[name] = value

    for field in _CONTEXT_FIELDS:
        context_window = _positive_int(entry.get(field))
        if context_window is not None:
            result["context_window"] = context_window
            break
    else:
        meta = entry.get("meta")
        if isinstance(meta, dict):
            result["context_window"] = _positive_int(meta.get("n_ctx_train"))

    result["function_calls"] = result["tools"]
    return result


def capabilities_from_llamacpp_props(props: Any) -> dict[str, bool | int | None]:
    """Read capabilities from a llama.cpp server ``GET /props`` response."""
    result = unknown_capabilities()
    if not isinstance(props, dict):
        return result

    caps = props.get("chat_template_caps")
    if isinstance(caps, dict) and caps:
        tool_flags = [
            caps[key]
            for key in ("supports_tools", "supports_tool_calls")
            if isinstance(caps.get(key), bool)
        ]
        if tool_flags:
            result["tools"] = all(tool_flags)
        if (
            caps.get("supports_reasoning_effort") is True
            or caps.get("supports_preserve_reasoning") is True
            or bool(caps.get("reasoning_effort_levels"))
        ):
            result["thinking"] = True

    modalities = props.get("modalities")
    if isinstance(modalities, dict):
        for flag in ("vision", "audio"):
            if isinstance(modalities.get(flag), bool):
                result[flag] = modalities[flag]

    settings = props.get("default_generation_settings")
    if isinstance(settings, dict):
        result["context_window"] = _positive_int(settings.get("n_ctx"))

    result["function_calls"] = result["tools"]
    return result


def find_model_entry(body: Any, model_name: str) -> dict | None:
    """Return the entry for ``model_name`` from a model-listing response."""
    if not isinstance(body, dict):
        return None
    entries = body.get("data", body.get("models"))
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and model_name in (
            entry.get("id"),
            entry.get("key"),
            entry.get("model_key"),
            entry.get("model"),
            entry.get("name"),
        ):
            return entry
    return None


def _server_root(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url


def _get_json(client: Any, url: str) -> Any:
    try:
        response = client.get(url)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ConnectionError, OSError) as exc:
        log_event(
            RUN_LOGGER,
            "provider.capability_probe_failed",
            level=logging.DEBUG,
            error_type=type(exc).__name__,
        )
        return None


def detect_openai_compatible_capabilities(
    model_name: str,
    *,
    base_url: str,
    client_factory: Callable[..., Any],
    capability_cache: dict | None = None,
    max_cache_entries: int = 64,
    overrides: dict | None = None,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> dict[str, bool | int | None]:
    """Detect what a model on an OpenAI-compatible endpoint can do.

    Probe failures are logged and treated as "unknown"; this never raises.
    """
    cache_key = ("openai-compatible", base_url.rstrip("/"), model_name)
    detected = capability_cache.get(cache_key) if capability_cache is not None else None
    if detected is None:
        sources: list[dict] = []
        root = _server_root(base_url)
        try:
            with client_factory(
                headers={}, timeout=timeout, allow_loopback=True
            ) as client:
                if _is_loopback_url(base_url):
                    props = _get_json(
                        client, f"{root}/props?model={quote(model_name, safe='')}"
                    )
                    if props is None:
                        props = _get_json(client, f"{root}/props")
                    sources.append(capabilities_from_llamacpp_props(props))
                    for path in _LMSTUDIO_MODEL_PATHS:
                        entry = find_model_entry(
                            _get_json(client, f"{root}{path}"), model_name
                        )
                        if entry is not None:
                            sources.append(capabilities_from_model_entry(entry))
                            break
                listing = _get_json(client, f"{base_url.rstrip('/')}/models")
                sources.append(
                    capabilities_from_model_entry(find_model_entry(listing, model_name))
                )
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            log_event(
                RUN_LOGGER,
                "provider.capability_detection_failed",
                level=logging.WARNING,
                model=model_name,
                error_type=type(exc).__name__,
                exc_info=True,
            )
        # Server-reported values win; the family table only fills the gaps.
        detected = merge_capabilities(*sources, capabilities_from_family(model_name))
        if capability_cache is not None:
            if len(capability_cache) >= max_cache_entries:
                capability_cache.clear()
            capability_cache[cache_key] = detected.copy()
    return merge_capabilities(overrides, detected)
