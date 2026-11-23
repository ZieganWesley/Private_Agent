"""Keyword retrieval uses SQLite FTS5 for large corpora (J9, 13.9)."""

import statistics
import time

from langchain_core.documents import Document

from rag import retrieval
from rag.retrieval import HybridRAGRetriever


def _documents(count: int) -> list[Document]:
    return [
        Document(
            page_content=(
                f"chunk number {index} about topic{index % 97} and keyword{index % 13}"
            ),
            metadata={"source": f"doc-{index}.md", "chunk": index},
        )
        for index in range(count)
    ]


def test_small_corpus_still_uses_bm25(monkeypatch):
    monkeypatch.setattr(retrieval, "_FTS5_MIN_DOCUMENTS", 10_000)
    retriever = HybridRAGRetriever(None, _documents(10))
    assert retriever.keyword_backend == "bm25"


def test_large_corpus_uses_fts5(monkeypatch):
    monkeypatch.setattr(retrieval, "_FTS5_MIN_DOCUMENTS", 100)
    retriever = HybridRAGRetriever(None, _documents(200))
    assert retriever.keyword_backend == "fts5"
    assert retriever.bm25 is None


def test_fts5_ranks_matching_chunks(monkeypatch):
    monkeypatch.setattr(retrieval, "_FTS5_MIN_DOCUMENTS", 100)
    documents = _documents(200)
    retriever = HybridRAGRetriever(None, documents)
    results = retriever._keyword_results(["topic42"], 5)
    assert results
    assert all("topic42" in doc.page_content for doc in results)


def test_fts5_quotes_query_terms(monkeypatch):
    monkeypatch.setattr(retrieval, "_FTS5_MIN_DOCUMENTS", 10)
    documents = [
        Document(page_content=f"alpha {index} γ-version don't", metadata={})
        for index in range(50)
    ]
    retriever = HybridRAGRetriever(None, documents)
    assert retriever.keyword_backend == "fts5"
    # Hyphens and apostrophes are legal inside quoted FTS5 phrases.
    assert retriever._keyword_results(["γ-version", "don't"], 3)


def test_falls_back_to_bm25_when_fts5_is_unavailable(monkeypatch):
    monkeypatch.setattr(retrieval, "_FTS5_MIN_DOCUMENTS", 100)
    monkeypatch.setattr(
        retrieval._Fts5KeywordIndex, "available", staticmethod(lambda: False)
    )
    retriever = HybridRAGRetriever(None, _documents(200))
    assert retriever.keyword_backend == "bm25"
    assert retriever._keyword_results(["topic42"], 5)


def test_keyword_search_at_twenty_thousand_chunks_stays_fast():
    documents = _documents(20_000)
    retriever = HybridRAGRetriever(None, documents)
    assert retriever.keyword_backend == "fts5"
    retriever._keyword_results(["topic42"], 5)  # build the index once
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        assert retriever._keyword_results(["topic42", "keyword3"], 5)
        timings.append(time.perf_counter() - started)
    assert statistics.median(timings) < 0.010
