"""Migration helpers for legacy endpoint configuration keys.

This module deliberately imports nothing from the application so both
``config`` and ``providers.endpoints`` can use it without a circular import.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VALID_ROLES = ("chat", "embedding", "rerank", "native")


def with_v1(url: str) -> str:
    """Append ``/v1`` to a legacy server URL that has no API prefix."""
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def fold_legacy_ollama_endpoints(models: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror ``ollama_base_url(s)`` into ``local_openai_compatible_endpoints``.

    The local endpoint list is the single description of locally hosted servers.
    A legacy Ollama URL is therefore materialized as an ordinary entry so both
    representations stay in sync without a second discovery path.
    """
    if not isinstance(models, Mapping):
        return {}
    result = dict(models)
    legacy = models.get("ollama_base_urls", models.get("ollama_base_url"))
    if isinstance(legacy, str):
        legacy = [legacy]
    if not isinstance(legacy, list):
        return result
    urls = [
        value.strip() for value in legacy if isinstance(value, str) and value.strip()
    ]
    if not urls:
        return result

    raw_entries = models.get("local_openai_compatible_endpoints")
    entries = (
        [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
        if isinstance(raw_entries, list)
        else []
    )
    known = {str(entry.get("base_url", "")).rstrip("/").lower() for entry in entries}
    for index, url in enumerate(urls):
        normalized = with_v1(url).rstrip("/")
        if normalized.lower() in known:
            continue
        entries.append(
            {
                "name": "Ollama" if index == 0 else f"Ollama {index + 1}",
                "base_url": normalized,
                "roles": list(VALID_ROLES),
            }
        )
        known.add(normalized.lower())
    result["local_openai_compatible_endpoints"] = entries
    return result


__all__ = ["VALID_ROLES", "fold_legacy_ollama_endpoints", "with_v1"]
