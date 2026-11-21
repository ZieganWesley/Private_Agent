"""Hybrid vector and lexical document retrieval."""

import heapq
import re
import sqlite3
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document

from private_agent.console import console

from ..config import RAG_RELEVANCE_THRESHOLD, RAG_SIMILARITY_RESULTS

_RRF_K = 60
# Above this many chunks, SQLite FTS5 ranks keyword matches with an index
# instead of scoring every chunk with BM25 (J9, step 13.9).
_FTS5_MIN_DOCUMENTS = 5_000
_TOKEN_PATTERN = re.compile(r"\w+(?:[.'’\-]\w+)*", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase text and split it into word-like tokens without punctuation."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def _document_key(document: Document) -> tuple[str, str]:
    metadata = getattr(document, "metadata", None) or {}
    content = getattr(document, "page_content", "") or ""
    source = metadata.get("source", "")
    chunk = metadata.get("chunk")
    if chunk is None:
        chunk = content[:30]
    return str(source), str(chunk)


def _reciprocal_rank_fusion(
    ranked_lists: tuple[list[Document], ...],
    k: int,
    *,
    rrf_k: int = _RRF_K,
) -> list[Document]:
    """Fuse ranked result lists, deduplicating by source and chunk."""
    scores: dict[tuple[str, str], float] = {}
    documents: dict[tuple[str, str], Document] = {}
    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            key = _document_key(document)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ordered_keys = sorted(documents, key=lambda key: -scores[key])
    return [documents[key] for key in ordered_keys[:k]]


class _Fts5KeywordIndex:
    """Rank chunk text with SQLite FTS5 instead of scoring the whole corpus."""

    def __init__(self, documents: list[Document]):
        self._documents = documents
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.execute("CREATE VIRTUAL TABLE chunks USING fts5(body)")
        self._connection.executemany(
            "INSERT INTO chunks(rowid, body) VALUES (?, ?)",
            (
                (index, doc.page_content or "")
                for index, doc in enumerate(documents, start=1)
            ),
        )
        self._connection.commit()

    @staticmethod
    def available() -> bool:
        try:
            connection = sqlite3.connect(":memory:")
            try:
                connection.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
            finally:
                connection.close()
        except sqlite3.Error:
            return False
        return True

    def search(self, tokens: list[str], k: int) -> list[Document]:
        # Quoting each term keeps FTS5 syntax characters out of the query.
        query = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        if not query:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT rowid FROM chunks WHERE chunks MATCH ? "
                "ORDER BY bm25(chunks) LIMIT ?",
                (query, k),
            ).fetchall()
        results = []
        for (rowid,) in rows:
            index = int(rowid) - 1
            if 0 <= index < len(self._documents):
                results.append(self._documents[index])
        return results


class HybridRAGRetriever:
    """Combine Chroma vector similarity and lexical retrieval.

    Small corpora use BM25 in memory; large corpora use a SQLite FTS5 index
    built lazily on the first keyword query (J9).
    """

    bm25 = None
    keyword_backend: str | None = None
    _fts5: _Fts5KeywordIndex | None = None

    def __init__(self, vectorstore: Chroma, documents: list[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = None
        self.keyword_backend: str | None = None
        self._fts5: _Fts5KeywordIndex | None = None
        if len(documents) >= _FTS5_MIN_DOCUMENTS and _Fts5KeywordIndex.available():
            self.keyword_backend = "fts5"
            return

        try:
            from rank_bm25 import BM25Okapi

            if documents:
                tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
                if any(tokenized_corpus):
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    self.keyword_backend = "bm25"
        except ImportError:
            console.print(
                "[yellow][Warning] 'rank_bm25' package not found. Falling back "
                "to pure vector similarity search.[/yellow]"
            )

    def _active_keyword_backend(self) -> str | None:
        """Infer BM25 when a caller built the retriever without ``__init__``."""
        if self.keyword_backend:
            return self.keyword_backend
        return "bm25" if self.bm25 else None

    def _keyword_results(
        self, query_tokens: list[str], k: int
    ) -> list[Document]:
        """Return the best keyword matches from FTS5 or BM25."""
        if self._active_keyword_backend() == "fts5":
            if self._fts5 is None:
                self._fts5 = _Fts5KeywordIndex(self.documents)
            return self._fts5.search(query_tokens, k)
        if not self.bm25:
            return []
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_scored_docs = heapq.nsmallest(
            k,
            enumerate(bm25_scores),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            self.documents[index] for index, score in top_scored_docs if score > 0
        ]

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Re-rank documents by embedding similarity when an embedder is available."""
        if not documents:
            return documents
        embeddings = getattr(self.vectorstore, "embeddings", None)
        if embeddings is None or not hasattr(embeddings, "embed_query"):
            return documents
        try:
            query_vec = embeddings.embed_query(query)
            doc_vecs = embeddings.embed_documents(
                [getattr(d, "page_content", "") or "" for d in documents]
            )
        except (AttributeError, ValueError, RuntimeError, TypeError):
            return documents
        if not query_vec or not doc_vecs or len(doc_vecs) != len(documents):
            return documents

        def _cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = sorted(
            zip(documents, doc_vecs, strict=False),
            key=lambda item: _cosine(query_vec, item[1]),
            reverse=True,
        )
        return [doc for doc, _ in scored]

    def similarity_search(
        self,
        query: str,
        k: int = RAG_SIMILARITY_RESULTS,
        collection: str | None = None,
    ) -> list[Document]:
        threshold = RAG_RELEVANCE_THRESHOLD
        vector_results = None
        scored_method = getattr(
            self.vectorstore, "similarity_search_with_relevance_scores", None
        )
        if callable(scored_method):
            try:
                scored = scored_method(query, k=k)
            except Exception:  # noqa: BLE001 - backend-specific failure
                scored = None
            if (
                isinstance(scored, list)
                and scored
                and all(isinstance(item, tuple) and len(item) == 2 for item in scored)
            ):
                vector_results = [doc for doc, score in scored if score >= threshold]
                if not vector_results:
                    return []
        if vector_results is None:
            vector_results = self.vectorstore.similarity_search(query, k=k)
            if threshold > 0:
                # No scores available; keep results to preserve behaviour.
                pass
        if collection:
            vector_results = [
                doc
                for doc in vector_results
                if (getattr(doc, "metadata", None) or {}).get("collection")
                == collection
            ]
            if not vector_results:
                return []
        query_tokens = _tokenize(query)
        if not self._active_keyword_backend() or not self.documents or not query_tokens:
            return self._rerank(query, vector_results[:k])

        keyword_results = self._keyword_results(query_tokens, k)
        if collection:
            keyword_results = [
                doc
                for doc in keyword_results
                if (getattr(doc, "metadata", None) or {}).get("collection")
                == collection
            ]

        fused = _reciprocal_rank_fusion((vector_results, keyword_results), k)
        return self._rerank(query, fused)
