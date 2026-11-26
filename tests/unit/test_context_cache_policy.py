"""C6: KV prefix policy gating, stability, and candidate bounding."""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from agent.context_cache import ContextCachePolicy
from agent.context_runtime import prepare_context_request
from rag.retrieval import HybridRAGRetriever


def _prepare(policy, user="hello", previous_hash=None, system="system rules " * 40):
    return prepare_context_request(
        system_prompts=[SystemMessage(content=system)],
        prompt_history=[HumanMessage(content="earlier turn")],
        user_message=HumanMessage(content=user),
        policy=policy,
        provider="openai-compatible",
        previous_hash=previous_hash,
    )


def test_disabled_policy_skips_prefix_hashing_and_hints():
    messages, meta = _prepare(ContextCachePolicy(enabled=False))

    assert [type(message).__name__ for message in messages] == [
        "SystemMessage",
        "HumanMessage",
        "HumanMessage",
    ]
    assert meta["cache"]["prefix_hash"] is None
    assert meta["cache"]["qualifies"] is False
    assert meta["cache"]["provider"]["enabled"] is False
    assert meta["telemetry"]["invalidations"] == 0


def test_volatile_tail_changes_do_not_invalidate_the_prefix():
    policy = ContextCachePolicy(enabled=True, min_prefix_tokens=10)
    _, first = _prepare(policy, user="hello")

    _, second = _prepare(
        policy,
        user="a completely different question",
        previous_hash=first["cache"]["prefix_hash"],
    )

    assert second["cache"]["prefix_hash"] == first["cache"]["prefix_hash"]
    assert second["cache"]["invalidated"] is False


def test_unsupported_provider_gets_no_hint_markers():
    policy = ContextCachePolicy(enabled=True, min_prefix_tokens=10)

    hint = policy.provider_hint("mystery-host")

    assert hint["enabled"] is False
    assert hint["markers"] == {}


def test_cacheable_sources_are_ordered_before_volatile_sources():
    policy = ContextCachePolicy(enabled=True)

    assert policy.stable_source_order(["rag", "system", "history", "user"]) == [
        "system",
        "rag",
        "history",
        "user",
    ]


class _RecordingReranker:
    def __init__(self):
        self.received = None

    def rerank(self, query, documents):
        self.received = list(documents)
        return list(documents)


def _doc(index, duplicate=False):
    return Document(
        page_content="duplicate text" if duplicate else f"text {index}",
        metadata={"source": "s", "chunk": "dup" if duplicate else index},
    )


def test_candidates_are_deduped_and_capped_before_reranking():
    retriever = object.__new__(HybridRAGRetriever)
    retriever.reranker = _RecordingReranker()
    retriever.vectorstore = None
    retriever.max_candidates = 5
    documents = [_doc(i) for i in range(6)] + [
        _doc(99, duplicate=True),
        _doc(99, duplicate=True),
    ]

    retriever._rerank("query", documents)

    assert len(retriever.reranker.received) == 5
    keys = [
        (doc.metadata.get("source"), doc.metadata.get("chunk"), doc.page_content)
        for doc in retriever.reranker.received
    ]
    assert len(keys) == len(set(keys))
