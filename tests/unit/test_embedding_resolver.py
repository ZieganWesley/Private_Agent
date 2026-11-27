"""Generic embedding and reranker route discovery."""

from types import SimpleNamespace

from rag.embedding_resolver import embedding_routes, rerank_routes


def _runtime(name, base_url, embeddings=(), rerankers=()):
    records = [SimpleNamespace(model_id=model, role="reranker") for model in rerankers]
    return SimpleNamespace(
        name=name,
        base_url=base_url,
        embedding_models=tuple(embeddings),
        records=tuple(records),
    )


def _endpoint(name, base_url, models=()):
    return SimpleNamespace(name=name, base_url=base_url, models=tuple(models))


class _Registry:
    def __init__(self, embedding=(), rerank=()):
        self._roles = {"embedding": embedding, "rerank": rerank}

    def for_role(self, role):
        return tuple(self._roles.get(role, ()))


def test_embedding_routes_come_from_runtimes_and_endpoints():
    routes = embedding_routes(
        [_runtime("local-a", "http://127.0.0.1:1111", ("embed-a", "embed-b"))],
        _Registry(embedding=[_endpoint("remote", "https://models.example/v1", ("e",))]),
    )

    assert [(route.label, route.model, route.base_url) for route in routes] == [
        ("local-a", "embed-a", "http://127.0.0.1:1111/v1"),
        ("local-a", "embed-b", "http://127.0.0.1:1111/v1"),
        ("remote", "e", "https://models.example/v1"),
    ]


def test_embedding_routes_dedupe_the_same_endpoint_and_model():
    routes = embedding_routes(
        [_runtime("local", "http://127.0.0.1:1111", ("embed",))],
        _Registry(embedding=[_endpoint("local", "http://127.0.0.1:1111", ("embed",))]),
    )

    assert len(routes) == 1


def test_rerank_routes_come_from_records_and_endpoints():
    routes = rerank_routes(
        [_runtime("local", "http://127.0.0.1:1111", rerankers=("rank-a",))],
        _Registry(
            rerank=[_endpoint("remote", "https://models.example/v1", ("rank-b",))]
        ),
    )

    assert [(route.label, route.model) for route in routes] == [
        ("local", "rank-a"),
        ("remote", "rank-b"),
    ]


def test_absent_capabilities_produce_no_routes():
    assert (
        embedding_routes([_runtime("local", "http://127.0.0.1:1111")], _Registry())
        == ()
    )
    assert (
        rerank_routes([_runtime("local", "http://127.0.0.1:1111")], _Registry()) == ()
    )
    assert embedding_routes([], None) == ()
    assert rerank_routes([], None) == ()
