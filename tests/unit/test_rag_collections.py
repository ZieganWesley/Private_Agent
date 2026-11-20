from unittest.mock import MagicMock

from private_agent.rag.retrieval import HybridRAGRetriever


def _doc(text, collection):
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"source": text, "collection": collection}
    return doc


def test_similarity_search_filters_by_collection():
    a = _doc("a", "docs")
    b = _doc("b", "notes")
    retriever = HybridRAGRetriever.__new__(HybridRAGRetriever)
    retriever.vectorstore = MagicMock()
    retriever.vectorstore.similarity_search.return_value = [a, b]
    retriever.bm25 = None
    retriever.documents = [a, b]
    result = retriever.similarity_search("q", k=5, collection="docs")
    assert result == [a]


def test_similarity_search_no_filter_returns_all():
    a = _doc("a", "docs")
    b = _doc("b", "notes")
    retriever = HybridRAGRetriever.__new__(HybridRAGRetriever)
    retriever.vectorstore = MagicMock()
    retriever.vectorstore.similarity_search.return_value = [a, b]
    retriever.bm25 = None
    retriever.documents = [a, b]
    result = retriever.similarity_search("q", k=5)
    assert result == [a, b]
