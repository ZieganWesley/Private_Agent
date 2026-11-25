"""Unit tests for optional /v1/rerank and the local cosine fallback."""

from dataclasses import dataclass

import httpx

from rag.rerank import (
    CircuitBreaker,
    HttpReranker,
    local_cosine_rerank,
    rerank_with_fallback,
    reranker_from_config,
)


@dataclass
class Doc:
    page_content: str


class FakeResponse:
    def __init__(self, body=None, error=None):
        self.body = body
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.body


class FakeClient:
    def __init__(self, responses, *, fail_after=None):
        self.responses = list(responses)
        self.calls = []
        self.closed = False
        self.fail_after = fail_after

    def post(self, url, json):
        self.calls.append((url, json))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise httpx.ConnectError("boom")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def _client_factory(client):
    return lambda **kwargs: client


def test_http_rerank_orders_by_score_and_respects_top_n():
    docs = [Doc("a"), Doc("b"), Doc("c")]
    client = FakeClient(
        [
            FakeResponse(
                {
                    "results": [
                        {"index": 2, "relevance_score": 0.1},
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": 0.5},
                    ]
                }
            )
        ]
    )
    reranker = HttpReranker(
        base_url="http://127.0.0.1:8080/v1",
        client_factory=_client_factory(client),
    )

    result = reranker.rerank("q", docs, top_n=2)

    assert [doc.page_content for doc in result] == ["a", "b"]
    assert client.calls[0][0] == "http://127.0.0.1:8080/v1/rerank"
    assert client.calls[0][1]["top_n"] == 2
    assert client.closed is True


def test_http_rerank_accepts_data_and_list_shapes():
    docs = [Doc("a"), Doc("b")]
    for body in (
        {"data": [{"index": 1, "score": 2.0}, {"index": 0, "score": 1.0}]},
        [{"index": 1, "score": 2.0}, {"index": 0, "score": 1.0}],
    ):
        reranker = HttpReranker(
            base_url="http://127.0.0.1:8080/v1",
            client_factory=_client_factory(FakeClient([FakeResponse(body)])),
        )
        assert [doc.page_content for doc in reranker.rerank("q", docs)] == ["b", "a"]


def test_http_rerank_returns_none_on_invalid_body():
    reranker = HttpReranker(
        base_url="http://127.0.0.1:8080/v1",
        client_factory=_client_factory(FakeClient([FakeResponse({"nope": []})])),
    )

    assert reranker.rerank("q", [Doc("a")]) is None


def test_http_rerank_caches_identical_requests():
    docs = [Doc("a"), Doc("b")]
    client = FakeClient(
        [FakeResponse({"results": [{"index": 1, "score": 1.0}]})]
    )
    reranker = HttpReranker(
        base_url="http://127.0.0.1:8080/v1",
        client_factory=_client_factory(client),
    )

    first = reranker.rerank("q", docs)
    second = reranker.rerank("q", docs)

    assert [doc.page_content for doc in first] == ["b"]
    assert [doc.page_content for doc in second] == ["b"]
    assert len(client.calls) == 1


def test_circuit_breaker_opens_after_threshold_and_cools_down():
    now = [100.0]
    breaker = CircuitBreaker(
        failure_threshold=2,
        cooldown_seconds=10,
        clock=lambda: now[0],
    )

    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.allow() is False
    now[0] += 11
    assert breaker.allow() is True
    assert breaker.failures == 0


def test_http_rerank_trips_breaker_after_repeated_failures():
    docs = [Doc("a")]
    client = FakeClient([], fail_after=0)
    reranker = HttpReranker(
        base_url="http://127.0.0.1:8080/v1",
        client_factory=_client_factory(client),
        breaker=CircuitBreaker(failure_threshold=2, cooldown_seconds=60),
    )

    assert reranker.rerank("q", docs) is None
    assert reranker.rerank("q", docs) is None
    assert reranker.rerank("q", docs) is None
    # The third call is rejected by the breaker, not sent.
    assert len(client.calls) == 2


class FakeEmbeddings:
    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        vectors = [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]]
        return vectors[: len(texts)]


def test_local_cosine_rerank_orders_by_similarity():
    docs = [Doc("orthogonal"), Doc("same"), Doc("diagonal")]

    result = local_cosine_rerank("query", docs, embeddings=FakeEmbeddings())

    assert [doc.page_content for doc in result] == ["same", "diagonal", "orthogonal"]


def test_rerank_with_fallback_uses_api_when_available():
    docs = [Doc("a"), Doc("b")]
    reranker = HttpReranker(
        base_url="http://127.0.0.1:8080/v1",
        client_factory=_client_factory(
            FakeClient([FakeResponse({"results": [{"index": 1, "score": 1.0}]})])
        ),
    )

    result = rerank_with_fallback(
        "q", docs, embeddings=FakeEmbeddings(), reranker=reranker
    )

    assert [doc.page_content for doc in result] == ["b"]


def test_reranker_from_config_uses_the_rerank_endpoint():
    reranker = reranker_from_config(
        {
            "models": {
                "endpoints": [
                    {
                        "name": "ranker",
                        "base_url": "http://127.0.0.1:9000/v1",
                        "roles": ["rerank"],
                        "models": ["bge-reranker"],
                    }
                ]
            },
            "rag": {"rerank_top_n": 5},
        }
    )

    assert reranker is not None
    assert reranker.base_url == "http://127.0.0.1:9000/v1"
    assert reranker.model == "bge-reranker"
    assert reranker.top_n == 5


def test_reranker_from_config_returns_none_without_rerank_role():
    assert (
        reranker_from_config(
            {
                "models": {
                    "endpoints": [
                        {
                            "name": "chat",
                            "base_url": "http://127.0.0.1:9000/v1",
                            "roles": ["chat"],
                        }
                    ]
                }
            }
        )
        is None
    )


def test_rerank_with_fallback_uses_local_when_api_fails():
    docs = [Doc("orthogonal"), Doc("same")]
    reranker = HttpReranker(
        base_url="http://127.0.0.1:8080/v1",
        client_factory=_client_factory(FakeClient([], fail_after=0)),
    )

    result = rerank_with_fallback(
        "q", docs, embeddings=FakeEmbeddings(), reranker=reranker
    )

    assert [doc.page_content for doc in result] == ["same", "orthogonal"]
