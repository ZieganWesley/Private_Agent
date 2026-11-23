"""Public RAG indexing and retrieval API."""

from __future__ import annotations

__all__ = [
    "HybridRAGRetriever",
    "format_retrieved_citations",
    "initialize_knowledge_base",
    "knowledge_base_signature",
    "reset_knowledge_base",
]


def __getattr__(name: str):
    if name in {
        "initialize_knowledge_base",
        "knowledge_base_signature",
        "reset_knowledge_base",
    }:
        from . import indexing

        return getattr(indexing, name)
    if name == "HybridRAGRetriever":
        from . import retrieval

        return retrieval.HybridRAGRetriever
    if name == "format_retrieved_citations":
        from . import citations

        return citations.format_retrieved_citations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
