"""Unit coverage for response-based local server identification (I1k)."""

from types import SimpleNamespace

from private_agent.providers.identify import identify_server


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def _client(routes):
    """Return a fake client answering exactly the recorded routes."""

    class FakeClient:
        def get(self, url):
            if url not in routes:
                raise RuntimeError(f"unexpected probe: {url}")
            body = routes[url]
            if callable(body):
                body = body()
            return _FakeResponse(body)

    return FakeClient()


def test_identification_uses_llamacpp_props_response():
    client = _client(
        {
            "http://127.0.0.1:8080/props": {
                "default_generation_settings": {"n_ctx": 4096},
                "model_path": "/models/gguf-model.gguf",
                "total_slots": 1,
            },
            "http://127.0.0.1:8080/v1/models": {"data": [{"id": "gguf-model"}]},
        }
    )

    assert identify_server(client, "http://127.0.0.1:8080/v1") == "llama.cpp"


def test_identification_uses_lm_studio_model_response():
    client = _client(
        {
            "http://127.0.0.1:1234/api/v1/models": {
                "models": [{"key": "qwen3-8b", "max_context_length": 32768}]
            },
            "http://127.0.0.1:1234/v1/models": {"data": [{"id": "qwen3-8b"}]},
        }
    )

    assert identify_server(client, "http://127.0.0.1:1234/v1") == "LM Studio"


def test_identification_falls_back_to_the_model_owner_label():
    client = _client(
        {
            "http://127.0.0.1:8000/v1/models": {
                "data": [{"id": "meta-llama/Llama-3-8B", "owned_by": "vllm"}]
            }
        }
    )

    assert identify_server(client, "http://127.0.0.1:8000/v1") == "vLLM"


def test_identification_returns_none_for_an_unknown_server():
    client = _client(
        {
            "http://127.0.0.1:9911/v1/models": {
                "data": [{"id": "mystery-model", "owned_by": "someone"}]
            }
        }
    )

    assert identify_server(client, "http://127.0.0.1:9911/v1") is None


def test_identification_never_raises_when_probes_fail():
    class ExplodingClient:
        def get(self, url):
            raise OSError(f"no route to {url}")

    assert identify_server(ExplodingClient(), "http://127.0.0.1:8080/v1") is None


def test_discovery_renames_a_runtime_from_its_responses(monkeypatch):
    from private_agent.agent import local_runtimes

    routes = {
        "http://127.0.0.1:8080/v1/models": {"data": [{"id": "gguf-model"}]},
        "http://127.0.0.1:8080/props": {
            "default_generation_settings": {"n_ctx": 4096}
        },
    }

    class FakeClient:
        is_closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            if url not in routes:
                raise RuntimeError(f"unexpected probe: {url}")
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: routes[url],
            )

    runtimes = local_runtimes.discover_local_runtimes(
        None,
        [{"name": "local engine", "base_url": "http://127.0.0.1:8080/v1"}],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert [(runtime.name, runtime.models) for runtime in runtimes] == [
        ("llama.cpp", ("gguf-model",))
    ]
