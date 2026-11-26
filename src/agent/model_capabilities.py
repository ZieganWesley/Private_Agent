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
from dataclasses import replace
from typing import Any
from urllib.parse import quote

import httpx

from utils.run_logging import RUN_LOGGER, log_event

from .local_runtimes import DISCOVERY_TIMEOUT_SECONDS, _is_loopback_url
from .model_catalog import (
    CAPABILITY_FLAGS,
    DEFAULT_CATALOG_TTL_SECONDS,
    INPUT_FIELDS,
    OUTPUT_FIELDS,
    PROVENANCE_DECLARED,
    PROVENANCE_DEFAULT,
    PROVENANCE_PROBE,
    ModelCatalog,
    ModelRecord,
    build_model_record,
    default_model_record,
    merge_flag_sources,
    modalities_from_entry,
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


def _entry_limit(entry: Any, fields: Iterable[str]) -> int | None:
    if not isinstance(entry, dict):
        return None
    for name in fields:
        value = _positive_int(entry.get(name))
        if value is not None:
            return value
    return None


def _record_with_declared_overrides(
    record: ModelRecord,
    overrides: dict | None,
) -> ModelRecord:
    """Apply configured overrides to a cached record without mutating it."""
    if not isinstance(overrides, dict) or not overrides:
        return record
    flags = dict(record.flags)
    provenance = dict(record.provenance)
    changed = False
    for name in CAPABILITY_FLAGS:
        value = overrides.get(name)
        if isinstance(value, bool):
            flags[name] = value
            provenance[name] = PROVENANCE_DECLARED
            changed = True
    context_length = _positive_int(overrides.get("context_window"))
    input_tokens = _positive_int(overrides.get("input_tokens"))
    output_tokens = _positive_int(overrides.get("output_tokens"))
    if context_length is not None:
        provenance["context_length"] = PROVENANCE_DECLARED
    if input_tokens is not None:
        provenance["input_tokens"] = PROVENANCE_DECLARED
    if output_tokens is not None:
        provenance["output_tokens"] = PROVENANCE_DECLARED
    if not changed and all(
        value is None for value in (context_length, input_tokens, output_tokens)
    ):
        return record
    return replace(
        record,
        flags=flags,
        provenance=provenance,
        context_length=(
            context_length if context_length is not None else record.context_length
        ),
        input_tokens=(
            input_tokens if input_tokens is not None else record.input_tokens
        ),
        output_tokens=(
            output_tokens if output_tokens is not None else record.output_tokens
        ),
    )


def _probe_model_record(
    model_name: str,
    *,
    base_url: str,
    client_factory: Callable[..., Any],
    timeout: float,
    ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
) -> ModelRecord:
    """Probe one endpoint and turn every source into a catalog record."""
    root = _server_root(base_url)
    props: Any = None
    lm_entry: dict | None = None
    listing_entry: dict | None = None
    try:
        with client_factory(headers={}, timeout=timeout, allow_loopback=True) as client:
            if _is_loopback_url(base_url):
                props = _get_json(
                    client, f"{root}/props?model={quote(model_name, safe='')}"
                )
                if props is None:
                    props = _get_json(client, f"{root}/props")
                for path in _LMSTUDIO_MODEL_PATHS:
                    lm_entry = find_model_entry(
                        _get_json(client, f"{root}{path}"), model_name
                    )
                    if lm_entry is not None:
                        break
            listing = _get_json(client, f"{base_url.rstrip('/')}/models")
            listing_entry = find_model_entry(listing, model_name)
    except (httpx.HTTPError, ConnectionError, OSError) as exc:
        log_event(
            RUN_LOGGER,
            "provider.capability_detection_failed",
            level=logging.WARNING,
            model=model_name,
            error_type=type(exc).__name__,
            exc_info=True,
        )

    props_caps = capabilities_from_llamacpp_props(props)
    lm_caps = capabilities_from_model_entry(lm_entry)
    listing_caps = capabilities_from_model_entry(listing_entry)
    family_caps = capabilities_from_family(model_name)

    # Server-reported values win; the family table only fills the gaps.
    flags, flag_provenance = merge_flag_sources(
        [
            (PROVENANCE_PROBE, props_caps),
            (PROVENANCE_PROBE, lm_caps),
            (PROVENANCE_PROBE, listing_caps),
            (PROVENANCE_DEFAULT, family_caps),
        ]
    )

    modalities = modalities_from_entry(props)
    if not modalities:
        modalities = modalities_from_entry(lm_entry)
    if not modalities:
        modalities = modalities_from_entry(listing_entry)

    return build_model_record(
        model_name,
        base_url=base_url,
        provider="openai-compatible",
        role_entries=(listing_entry, lm_entry, props),
        context_candidates=(
            (PROVENANCE_PROBE, props_caps.get("context_window")),
            (PROVENANCE_PROBE, lm_caps.get("context_window")),
            (PROVENANCE_PROBE, listing_caps.get("context_window")),
        ),
        input_candidates=(
            (PROVENANCE_PROBE, _entry_limit(lm_entry, INPUT_FIELDS)),
            (PROVENANCE_PROBE, _entry_limit(listing_entry, INPUT_FIELDS)),
        ),
        output_candidates=(
            (PROVENANCE_PROBE, _entry_limit(lm_entry, OUTPUT_FIELDS)),
            (PROVENANCE_PROBE, _entry_limit(listing_entry, OUTPUT_FIELDS)),
        ),
        modality_candidates=(
            (PROVENANCE_PROBE if modalities else PROVENANCE_DEFAULT, modalities),
        ),
        flags=flags,
        flag_provenance=flag_provenance,
        ttl_seconds=ttl_seconds,
    )


def detect_openai_compatible_model_record(
    model_name: str,
    *,
    base_url: str,
    client_factory: Callable[..., Any],
    catalog: ModelCatalog | None = None,
    overrides: dict | None = None,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> ModelRecord:
    """Return the catalog record for one model, probing at most once."""
    store = catalog if catalog is not None else ModelCatalog()
    record = store.lookup(
        model_name,
        base_url,
        probe=lambda: _probe_model_record(
            model_name,
            base_url=base_url,
            client_factory=client_factory,
            timeout=timeout,
            ttl_seconds=store.ttl_seconds,
        ),
    )
    if record is None:
        record = default_model_record(model_name, base_url=base_url)
    return _record_with_declared_overrides(record, overrides)


def capabilities_with_limits(record: ModelRecord) -> dict[str, Any]:
    """Return capabilities plus the listing limits and their provenance."""
    surface: dict[str, Any] = dict(record.capabilities())
    surface.update(
        {
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "modalities": list(record.modalities),
            "role": record.role,
            "provenance": dict(record.provenance),
        }
    )
    return surface


def detect_openai_compatible_capabilities(
    model_name: str,
    *,
    base_url: str,
    client_factory: Callable[..., Any],
    capability_cache: dict | None = None,
    max_cache_entries: int = 64,
    overrides: dict | None = None,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
    catalog: ModelCatalog | None = None,
) -> dict[str, bool | int | None]:
    """Detect what a model on an OpenAI-compatible endpoint can do.

    The catalog owns the record, limits and provenance; this wrapper keeps the
    historical capability dict for existing callers.  Probe failures are logged
    and treated as "unknown"; this never raises.
    """
    cache_key = ("openai-compatible", base_url.rstrip("/"), model_name)
    detected = capability_cache.get(cache_key) if capability_cache is not None else None
    if detected is None:
        record = detect_openai_compatible_model_record(
            model_name,
            base_url=base_url,
            client_factory=client_factory,
            catalog=catalog,
            timeout=timeout,
        )
        detected = record.capabilities()
        if capability_cache is not None:
            if len(capability_cache) >= max_cache_entries:
                capability_cache.clear()
            capability_cache[cache_key] = detected.copy()
    return merge_capabilities(overrides, detected)
