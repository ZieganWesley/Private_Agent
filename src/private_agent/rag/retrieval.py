"""Hybrid vector and lexical document retrieval."""

import heapq
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console

from ..config import RAG_SIMILARITY_RESULTS

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

    def similarity_search(
        self, query: str, k: int = RAG_SIMILARITY_RESULTS
    ) -> list[Document]:
        vector_results = self.vectorstore.similarity_search(query, k=k)
        query_tokens = _tokenize(query)
        if not self.bm25 or not self.documents or not query_tokens:
            return vector_results[:k]

        bm25_scores = self.bm25.get_scores(query_tokens)
        top_scored_docs = heapq.nsmallest(
            k,
            enumerate(bm25_scores),
            key=lambda item: (-item[1], item[0]),
        )
        bm25_results = [
            self.documents[index] for index, score in top_scored_docs if score > 0
        ]

        return _reciprocal_rank_fusion((vector_results, bm25_results), k)
