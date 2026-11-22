"""Identify which local server answers on a port from its responses (I1k)."""

from __future__ import annotations

import logging
from typing import Any

from ..utils.run_logging import RUN_LOGGER, log_event

# Model-list ownership labels servers put in their OpenAI-compatible payloads.
_OWNER_LABELS = {
    "vllm": "vLLM",
    "llama.cpp": "llama.cpp",
    "jan": "Jan",
    "mlx": "MLX-LM",
    "mlx-lm": "MLX-LM",
    "gpt4all": "GPT4All",
    "localai": "LocalAI",
    "koboldcpp": "KoboldCpp",
    "text-generation-webui": "text-generation-webui",
    "lmstudio": "LM Studio",
    "lm-studio": "LM Studio",
}

# Root endpoints that only one known server family answers successfully.
_ROOT_PROBES = (
    ("/props", lambda body: "llama.cpp" if isinstance(body, dict) else None),
    ("/api/v1/models", lambda body: "LM Studio" if isinstance(body, dict) else None),
    (
        "/api/v1/info",
        lambda body: "KoboldCpp" if isinstance(body, dict) else None,
    ),
    (
        "/api/extra/version",
        lambda body: "KoboldCpp"
        if isinstance(body, dict) and "koboldcpp" in str(body).lower()
        else None,
    ),
)


def _server_root(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url


def _owned_by_labels(body: Any) -> set[str]:
    if not isinstance(body, dict):
        return set()
    entries = body.get("data", body.get("models"))
    if not isinstance(entries, list):
        return set()
    return {
        str(entry.get("owned_by", "")).strip().lower()
        for entry in entries
        if isinstance(entry, dict) and entry.get("owned_by")
    }


def identify_server(client: Any, base_url: str) -> str | None:
    """Return the known server family answering at ``base_url``, if identifiable.

    Uses only the responses the server already gives: the OpenAI-compatible
    model list and the few root endpoints that are unique to one family. Any
    probe failure means "not this family", never an exception.
    """
    root = _server_root(base_url)
    for path, classify in _ROOT_PROBES:
        try:
            response = client.get(f"{root}{path}")
            response.raise_for_status()
            label = classify(response.json())
        except Exception as exc:  # noqa: BLE001 - identification is best effort
            log_event(
                RUN_LOGGER,
                "provider.identify_probe_failed",
                level=logging.DEBUG,
                path=path,
                error_type=type(exc).__name__,
            )
            continue
        if label:
            return label
    try:
        response = client.get(f"{base_url.rstrip('/')}/models")
        response.raise_for_status()
        owners = _owned_by_labels(response.json())
    except Exception as exc:  # noqa: BLE001 - identification is best effort
        log_event(
            RUN_LOGGER,
            "provider.identify_probe_failed",
            level=logging.DEBUG,
            path="/models",
            error_type=type(exc).__name__,
        )
        return None
    for owner in owners:
        if owner in _OWNER_LABELS:
            return _OWNER_LABELS[owner]
    return None
