from unittest.mock import MagicMock

from rag.retrieval import HybridRAGRetriever


def test_rerank_uses_embeddings_when_available():
    class FakeEmbeddings:
        def embed_query(self, query):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            # second doc is more similar
            return [[0.0, 1.0], [1.0, 0.0]]

    docs = [MagicMock(page_content="a"), MagicMock(page_content="b")]

    class Retriever(HybridRAGRetriever):
        pass

    retriever = HybridRAGRetriever.__new__(HybridRAGRetriever)
    retriever.vectorstore = MagicMock()
    retriever.vectorstore.embeddings = FakeEmbeddings()
    result = retriever._rerank("q", docs)
    assert result == [docs[1], docs[0]]


def test_rerank_falls_back_without_embeddings():
    docs = [MagicMock(page_content="a"), MagicMock(page_content="b")]
    retriever = HybridRAGRetriever.__new__(HybridRAGRetriever)
    retriever.vectorstore = MagicMock()
    retriever.vectorstore.embeddings = None
    assert retriever._rerank("q", docs) == docs
