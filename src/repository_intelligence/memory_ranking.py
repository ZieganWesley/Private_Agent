"""Typed memory classification and ranking helpers (M3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MEMORY_TYPES = (
    "working",
    "long_term",
    "episodic",
    "project",
    "architecture",
    "user",
)


def classify_memory(text: str, *, tags: list[str] | None = None) -> str:
    lowered = (text or "").lower()
    tags = [t.lower() for t in (tags or [])]
    if "architecture" in lowered or "architecture" in tags:
        return "architecture"
    if any(k in lowered for k in ("project", "repository", "codebase")):
        return "project"
    if any(k in lowered for k in ("user", "preference", "likes")):
        return "user"
    if any(k in lowered for k in ("episode", "session", "conversation")):
        return "episodic"
    if any(k in lowered for k in ("remember", "long-term", "fact")):
        return "long_term"
    return "working"


def _recency_score(created_at: Any) -> float:
    try:
        if isinstance(created_at, (int, float)):
            dt = datetime.fromtimestamp(float(created_at), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(created_at))
        age_days = (datetime.now(timezone.utc) - dt).days
        return max(0.0, 1.0 - (age_days / 30.0))
    except Exception:  # noqa: BLE001 - timestamp parsing may fail
        return 0.5


def rank_memories(
    memories: list[dict[str, Any]], query: str = ""
) -> list[dict[str, Any]]:
    query_terms = {t for t in (query or "").lower().split() if t}
    ranked = []
    for m in memories:
        text = str(m.get("text") or m.get("content") or "")
        lowered = text.lower()
        relevance = 0.0
        if query_terms:
            relevance = sum(1 for t in query_terms if t in lowered) / len(query_terms)
        frequency = float(m.get("frequency", 1.0))
        importance = float(m.get("importance", 0.5))
        confidence = float(m.get("confidence", 0.5))
        recency = _recency_score(m.get("created_at"))
        feedback = float(m.get("feedback", 0.0))
        score = (
            0.25 * relevance
            + 0.2 * frequency / max(frequency, 1.0)
            + 0.2 * importance
            + 0.15 * confidence
            + 0.15 * recency
            + 0.05 * feedback
        )
        ranked.append({**m, "_score": score})
    ranked.sort(key=lambda x: x["_score"], reverse=True)
    return ranked
