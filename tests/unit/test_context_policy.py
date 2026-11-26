"""Unit tests for RAG cutoff and compaction policy."""

from dataclasses import dataclass

from agent.context_policy import apply_rag_cutoff, decide_compaction


@dataclass
class Doc:
    page_content: str
    metadata: dict


def test_rag_cutoff_deduplicates_and_drops_empty():
    docs = [
        Doc("a", {"source": "s", "chunk": 1}),
        Doc("a", {"source": "s", "chunk": 1}),
        Doc("   ", {"source": "s", "chunk": 2}),
        Doc("b", {"source": "s", "chunk": 3}),
    ]

    result = apply_rag_cutoff(docs, budget_tokens=100)

    assert [doc.page_content for doc in result.documents] == ["a", "b"]
    assert result.use_rag is True


def test_rag_cutoff_stops_at_budget():
    docs = [Doc("x" * 40, {}), Doc("y" * 40, {})]

    result = apply_rag_cutoff(
        docs, budget_tokens=20, count_tokens=lambda text: len(text)
    )

    assert result.use_rag is False
    assert result.reason == "empty"


def test_rag_cutoff_respects_max_results():
    docs = [Doc("a", {"chunk": 1}), Doc("b", {"chunk": 2}), Doc("c", {"chunk": 3})]

    result = apply_rag_cutoff(docs, budget_tokens=100, max_results=2)

    assert [doc.page_content for doc in result.documents] == ["a", "b"]


def test_rag_cutoff_rejects_zero_budget():
    result = apply_rag_cutoff([Doc("a", {})], budget_tokens=0)

    assert result.use_rag is False
    assert result.reason == "no_budget"


def test_rag_cutoff_requires_minimum_results():
    result = apply_rag_cutoff([Doc("a", {})], budget_tokens=100, min_results=2)

    assert result.use_rag is False
    assert result.reason == "below_min_results"


def test_compaction_uses_token_threshold():
    decision = decide_compaction(
        history_tokens=90,
        prompt_budget=100,
        threshold=0.8,
        message_count=3,
        max_messages=10,
    )

    assert decision.compact is True
    assert decision.reason == "token_threshold"


def test_compaction_uses_message_limit():
    decision = decide_compaction(
        history_tokens=10,
        prompt_budget=100,
        threshold=0.8,
        message_count=11,
        max_messages=10,
    )

    assert decision.compact is True
    assert decision.reason == "message_limit"


def test_compaction_skips_without_history():
    decision = decide_compaction(
        history_tokens=0,
        prompt_budget=100,
        threshold=0.8,
        message_count=0,
        max_messages=10,
    )

    assert decision.compact is False
    assert decision.reason == "no_history"
