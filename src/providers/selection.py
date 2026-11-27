"""Provider selection helpers: remembered choice, flags and installed-but-idle providers."""

from __future__ import annotations

import json
import logging
import os
import pathlib
from collections.abc import Iterable
from typing import Any

from utils.run_logging import RUN_LOGGER, log_event

STATE_ENV = "PRIVATE_AGENT_PROVIDER_STATE"
PROVIDER_ENV = "PRIVATE_AGENT_PROVIDER"
MODEL_ENV = "PRIVATE_AGENT_MODEL"
WORKSPACE_ENV = "PRIVATE_AGENT_WORKSPACE"
RAG_ENV = "PRIVATE_AGENT_RAG"
MODE_ENV = "PRIVATE_AGENT_MODE"
PERMISSION_MODE_ENV = "PRIVATE_AGENT_PERMISSION_MODE"


def state_path() -> pathlib.Path:
    """Return the file that remembers the last provider and model."""
    override = os.environ.get(STATE_ENV)
    if override:
        return pathlib.Path(override).expanduser()
    return pathlib.Path.home() / ".config" / "private_agent" / "last_provider.json"


def load_last_provider() -> dict[str, Any] | None:
    """Read the remembered provider choice, or None when absent or unreadable."""
    path = state_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _write_state(payload: dict[str, Any]) -> None:
    """Write the remembered state, preserving every section (best effort)."""
    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        log_event(
            RUN_LOGGER,
            "provider.last_choice_write_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
        )


def _load_route(key: str) -> dict[str, str] | None:
    """Read one remembered model route, or None when absent or malformed."""
    state = load_last_provider() or {}
    route = state.get(key)
    if not isinstance(route, dict):
        return None
    values = {}
    for field in ("provider", "model", "base_url"):
        value = route.get(field, "")
        values[field] = value.strip() if isinstance(value, str) else ""
    if not values["provider"] or not values["model"]:
        return None
    return values


def save_last_provider(
    *,
    adapter_type: str,
    display_name: str,
    base_url: str,
    model: str,
) -> None:
    """Remember the provider and model used by this session (best effort)."""
    state = load_last_provider() or {}
    state.update(
        {
            "adapter_type": str(adapter_type),
            "display_name": str(display_name),
            "base_url": str(base_url),
            "model": str(model),
        }
    )
    _write_state(state)


def load_embedding_choice() -> dict[str, str] | None:
    """Read the remembered RAG embedding route, if one was saved."""
    return _load_route("embedding")


def save_embedding_choice(*, provider: str, model: str, base_url: str) -> None:
    """Remember the RAG embedding route so auto-selection stays stable."""
    state = load_last_provider() or {}
    state["embedding"] = {
        "provider": str(provider),
        "model": str(model),
        "base_url": str(base_url),
    }
    _write_state(state)


def load_reranker_choice() -> dict[str, str] | None:
    """Read the remembered RAG reranker route, if one was saved."""
    return _load_route("reranker")


def save_reranker_choice(*, label: str, model: str, base_url: str) -> None:
    """Remember the selected reranker endpoint and model."""
    state = load_last_provider() or {}
    state["reranker"] = {
        "provider": str(label),
        "model": str(model),
        "base_url": str(base_url),
    }
    _write_state(state)


OLLAMA_EXTRA_HINT = (
    "Ollama support is an optional extra: `pip install 'private-agent[ollama]'`, "
    "or choose another runtime."
)


def optional_dependency_hint(kind: str, error: BaseException) -> str | None:
    """Return an install hint when an optional provider package is missing."""
    if isinstance(error, ImportError) and str(kind) == "ollama":
        return OLLAMA_EXTRA_HINT
    return None


def requested_provider() -> str:
    """Return the provider name requested on the command line, if any."""
    return os.environ.get(PROVIDER_ENV, "").strip()


def requested_model() -> str:
    """Return the model requested on the command line, if any."""
    return os.environ.get(MODEL_ENV, "").strip()


def requested_workspace() -> str:
    """Return the workspace root requested on the command line, if any."""
    return os.environ.get(WORKSPACE_ENV, "").strip()


def requested_rag() -> str:
    """Return the RAG documents directory requested on the command line."""
    return os.environ.get(RAG_ENV, "").strip()


def requested_mode() -> str:
    """Return the permission mode requested on the command line, if any."""
    return (
        os.environ.get(MODE_ENV, "").strip()
        or os.environ.get(PERMISSION_MODE_ENV, "").strip()
    )


def match_runtime(name: str, runtimes: Iterable[Any]) -> Any | None:
    """Return the reachable runtime whose name matches, case-insensitively."""
    wanted = str(name or "").strip().lower()
    if not wanted:
        return None
    for runtime in runtimes:
        if str(getattr(runtime, "name", "")).strip().lower() == wanted:
            return runtime
    return None


def pick_default_model(models: Iterable[str], requested: str = "") -> str | None:
    """Choose a model: the exact request, then the closest name match, then the first."""
    models = [str(model) for model in models if str(model).strip()]
    if not models:
        return None
    wanted = str(requested or "").strip().lower()
    if not wanted:
        return models[0]
    for model in models:
        if model.strip().lower() == wanted:
            return model
    for model in models:
        if wanted in model.lower():
            return model
    for model in models:
        if model.lower() in wanted:
            return model
    return models[0]
