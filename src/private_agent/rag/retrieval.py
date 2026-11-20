"""Hybrid vector and lexical document retrieval."""

import heapq
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console

from ..config import RAG_RELEVANCE_THRESHOLD, RAG_SIMILARITY_RESULTS

console = Console()

_RRF_K = 60
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


class HybridRAGRetriever:
    """Combine Chroma vector similarity and BM25 lexical retrieval."""

    def __init__(self, vectorstore: Chroma, documents: list[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = None

        try:
            from rank_bm25 import BM25Okapi

            if documents:
                tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
                if any(tokenized_corpus):
                    self.bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            console.print(
                "[yellow][Warning] 'rank_bm25' package not found. Falling back "
                "to pure vector similarity search.[/yellow]"
            )

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
            if isinstance(scored, list) and scored and all(
                isinstance(item, tuple) and len(item) == 2 for item in scored
            ):
                vector_results = [
                    doc for doc, score in scored if score >= threshold
                ]
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
        if not self.bm25 or not self.documents or not query_tokens:
            return self._rerank(query, vector_results[:k])

        bm25_scores = self.bm25.get_scores(query_tokens)
        top_scored_docs = heapq.nsmallest(
            k,
            enumerate(bm25_scores),
            key=lambda item: (-item[1], item[0]),
        )
        bm25_results = [
            self.documents[index] for index, score in top_scored_docs if score > 0
        ]
        if collection:
            bm25_results = [
                doc
                for doc in bm25_results
                if (getattr(doc, "metadata", None) or {}).get("collection")
                == collection
            ]

        fused = _reciprocal_rank_fusion((vector_results, bm25_results), k)
        return self._rerank(query, fused)
