"""C3: RAG relevance cutoff, budget-derived depth, and dedupe."""

import asyncio

from langchain_core.documents import Document

from agent import memory_context
from agent.memory_context import (
    build_rag_context,
    dedupe_rag_documents,
    filter_weak_rag_documents,
    rag_result_limit,
    trim_rag_documents_to_budget,
)


def _doc(text="content", score=None, source="doc.md", chunk=0):
    metadata = {"source": source, "chunk": chunk}
    if score is not None:
        metadata["relevance_score"] = score
    return Document(page_content=text, metadata=metadata)


def test_weak_scores_are_dropped_and_strong_scores_kept():
    documents = [_doc(score=0.1), _doc(score=0.9), _doc("no score")]

    kept = filter_weak_rag_documents(documents, relevance_threshold=0.25)

    assert [doc.metadata["relevance_score"] for doc in kept if "relevance_score" in doc.metadata] == [0.9]
    assert kept[-1].page_content == "no score"


def test_empty_documents_are_dropped():
    assert filter_weak_rag_documents([_doc("   "), _doc("ok")]) == [_doc("ok")]


def test_unscored_documents_keep_their_retrieval_order():
    documents = [_doc("first", chunk=1), _doc("second", chunk=2)]

    assert filter_weak_rag_documents(documents) == documents


def test_rag_result_limit_is_derived_from_budget_and_bounded():
    assert rag_result_limit(None) == memory_context.RAG_CONTEXT_RESULTS
    assert rag_result_limit(0) == 0
    assert rag_result_limit(10) == memory_context.RAG_MIN_RESULTS
    assert rag_result_limit(10_000_000) == memory_context.RAG_MAX_RESULTS
    assert rag_result_limit(800) < rag_result_limit(3200)


def test_dedupe_by_source_chunk_and_content():
    documents = [
        _doc("same", source="a", chunk=0),
        _doc("same", source="a", chunk=0),
        _doc("same", source="a", chunk=1),
    ]

    kept = dedupe_rag_documents(documents)

    assert len(kept) == 2


def test_budget_exhaustion_stops_without_partial_chunks():
    documents = [_doc("first " * 10), _doc("second " * 200), _doc("third " * 10)]
    budget = sum(
        memory_context._token_count(document.page_content) for document in documents[:2]
    )

    kept = trim_rag_documents_to_budget(documents, budget)

    assert kept == documents[:2]


def test_build_rag_context_returns_nothing_for_empty_or_weak_results():
    assert build_rag_context([], budget_tokens=100) == ("", [])
    weak = [_doc(score=0.01), _doc("   ")]
    assert build_rag_context(weak, budget_tokens=100) == ("", [])


def test_build_rag_context_never_exceeds_the_budget_including_headers():
    documents = [_doc("word " * 15, chunk=index) for index in range(4)]
    budget = 50

    text, kept = build_rag_context(documents, budget_tokens=budget)

    assert kept
    assert memory_context._token_count(text) <= budget
    assert len(kept) < len(documents)


def test_build_rag_context_formats_sources():
    text, kept = build_rag_context(
        [_doc("fact", source="manual.md", chunk=3)], budget_tokens=100
    )

    assert kept
    assert "[Source: manual.md, chunk 3]" in text
    assert "fact" in text


class _RecordingStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search(self, query, k=5):
        self.calls.append((query, k))
        return list(self.results)[:k]


def test_retrieval_uses_budget_derived_k_and_trims(monkeypatch):
    monkeypatch.setattr(memory_context, "RAG_INCLUDE_CODING_GUIDANCE", False)
    monkeypatch.setattr(memory_context, "_LAST_RAG_QUERY", "")
    documents = [_doc("word " * 30, chunk=index) for index in range(10)]
    store = _RecordingStore(documents)

    results = asyncio.run(
        memory_context._retrieve_rag_documents(
            store,
            "query",
            include_coding_guidance=False,
            budget_tokens=800,
        )
    )

    expected_k = rag_result_limit(800)
    assert store.calls == [("query", expected_k)]
    assert results
    assert sum(
        memory_context._token_count(doc.page_content) for doc in results
    ) <= 800


def test_retrieval_returns_nothing_when_budget_is_exhausted():
    store = _RecordingStore([_doc("anything")])

    results = asyncio.run(
        memory_context._retrieve_rag_documents(
            store,
            "query",
            include_coding_guidance=False,
            budget_tokens=0,
        )
    )

    assert results == []
    assert store.calls == []


def test_retrieval_dedupes_before_scoring():
    duplicates = [_doc("same", chunk=1), _doc("same", chunk=1), _doc(score=0.01)]
    store = _RecordingStore(duplicates)

    results = asyncio.run(
        memory_context._retrieve_rag_documents(
            store,
            "query",
            include_coding_guidance=False,
            budget_tokens=1000,
        )
    )

    assert results == [_doc("same", chunk=1)]
