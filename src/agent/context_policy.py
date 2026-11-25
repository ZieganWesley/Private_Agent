"""Small, testable policies for RAG cutoff and history compaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _default_count(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _content(document: Any) -> str:
    text = getattr(document, "page_content", None)
    if text is None:
        text = getattr(document, "content", "")
    return str(text or "")


def _key(document: Any) -> tuple[str, str, str]:
    metadata = getattr(document, "metadata", None) or {}
    return (
        str(metadata.get("source", "")),
        str(metadata.get("chunk", "")),
        _content(document),
    )


@dataclass
class RagCutoff:
    use_rag: bool
    documents: list[Any] = field(default_factory=list)
    tokens: int = 0
    reason: str = ""


def apply_rag_cutoff(
    documents: Sequence[Any] | None,
    *,
    budget_tokens: int | None,
    max_results: int | None = None,
    min_results: int = 1,
    count_tokens: Callable[[str], int] | None = None,
) -> RagCutoff:
    """Deduplicate, drop empty chunks, and stop at the RAG budget."""
    count = count_tokens or _default_count
    if budget_tokens is not None and budget_tokens <= 0:
        return RagCutoff(False, [], 0, "no_budget")
    seen: set[tuple[str, str, str]] = set()
    kept: list[Any] = []
    used = 0
    for document in documents or []:
        text = _content(document)
        if not text.strip():
            continue
        key = _key(document)
        if key in seen:
            continue
        seen.add(key)
        cost = max(0, int(count(text)))
        if budget_tokens is not None and used + cost > budget_tokens:
            break
        kept.append(document)
        used += cost
        if max_results is not None and len(kept) >= max(1, int(max_results)):
            break
    if not kept:
        return RagCutoff(False, [], 0, "empty")
    if len(kept) < max(1, int(min_results)):
        return RagCutoff(False, [], used, "below_min_results")
    return RagCutoff(True, kept, used, "ok")


@dataclass(frozen=True)
class CompactionDecision:
    compact: bool
    reason: str
    ratio: float
    message_count: int


def decide_compaction(
    *,
    history_tokens: int,
    prompt_budget: int,
    threshold: float,
    message_count: int,
    max_messages: int,
) -> CompactionDecision:
    """Compact when the token ratio or message count crosses its limit."""
    budget = max(1, int(prompt_budget))
    ratio = max(0.0, float(history_tokens)) / budget
    if message_count <= 0:
        return CompactionDecision(False, "no_history", ratio, message_count)
    if ratio >= max(0.0, float(threshold)):
        return CompactionDecision(True, "token_threshold", ratio, message_count)
    if message_count > max(1, int(max_messages)):
        return CompactionDecision(True, "message_limit", ratio, message_count)
    return CompactionDecision(False, "within_budget", ratio, message_count)


__all__ = [
    "CompactionDecision",
    "RagCutoff",
    "apply_rag_cutoff",
    "decide_compaction",
]
