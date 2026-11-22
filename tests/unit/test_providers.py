from agent.providers import create_local_chat_model


def test_local_model_receives_bounded_context_and_output_options():
    captured = {}

    def model_factory(**options):
        captured.update(options)
        return object()

    create_local_chat_model(
        "local-model",
        temperature=0.1,
        base_url="http://localhost:11434",
        acceleration_mode="auto",
        thinking=None,
        chat_model_factory=model_factory,
        client_kwargs=dict,
        acceleration_options=lambda _mode: {},
        track_clients=lambda _model: None,
        max_output_tokens=512,
        context_window=4096,
    )

    assert captured["num_ctx"] == 4096
    assert captured["num_predict"] == 512


def test_openai_compatible_model_receives_injected_policy_clients():
    from agent.providers import create_openai_compatible_chat_model

    captured = {}
    sync_client = object()
    async_client = object()

    def model_factory(**options):
        captured.update(options)
        return "model"

    def sync_client_factory(**options):
        captured["sync_client_options"] = options
        return sync_client

    def async_client_factory(**options):
        captured["async_client_options"] = options
        return async_client

    model = create_openai_compatible_chat_model(
        "https://api.example.test/v1",
        "test-model",
        "session-key",
        temperature=0.2,
        request_timeout=45,
        max_retries=2,
        max_tokens=1024,
        chat_model_factory=model_factory,
        sync_client_factory=sync_client_factory,
        async_client_factory=async_client_factory,
    )

    assert model == "model"
    assert captured["model"] == "test-model"
    assert captured["base_url"] == "https://api.example.test/v1"
    assert captured["api_key"] == "session-key"
    assert captured["temperature"] == 0.2
    assert captured["timeout"] == 45
    assert captured["max_retries"] == 2
    assert captured["max_tokens"] == 1024
    assert captured["http_client"] is sync_client
    assert captured["http_async_client"] is async_client
    assert captured["sync_client_options"] == {
        "headers": {},
        "timeout": 45,
        "allow_loopback": True,
    }
    assert captured["async_client_options"] == {
        "headers": {},
        "timeout": 45,
        "allow_loopback": True,
        "track": True,
    }


def test_online_capabilities_use_model_profile_without_guessing():
    from types import SimpleNamespace

    from agent.providers import online_capabilities

    known = SimpleNamespace(
        profile={
            "tool_calling": True,
            "structured_output": True,
            "image_inputs": True,
            "audio_inputs": False,
            "reasoning_output": False,
        }
    )
    assert online_capabilities(known, True) == {
        "tools": True,
        "function_calls": True,
        "structured_output": True,
        "thinking": False,
        "vision": True,
        "audio": False,
    }
    disabled = online_capabilities(known, False)
    assert disabled["tools"] is False and disabled["function_calls"] is False
    unknown = online_capabilities(SimpleNamespace(profile={}), True)
    assert all(value is None for value in unknown.values())
    assert online_capabilities(object(), True)["vision"] is None


def test_local_runtime_discovery_reports_only_reachable_loopback_runtimes():
    from types import SimpleNamespace

    from agent.local_runtimes import discover_local_runtimes

    responses = {
        "http://127.0.0.1:11434/api/tags": {
            "models": [
                {"name": "qwen:latest"},
                {"name": "nomic-embed-text"},
            ]
        },
        "http://127.0.0.1:1234/v1/models": {"data": [{"id": "lmstudio-model"}]},
        "http://127.0.0.1:8080/v1/models": {"data": [{"id": "gguf-model"}]},
    }

    class FakeClient:
        is_closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: responses[url],
            )

    runtimes = discover_local_runtimes(
        "http://127.0.0.1:11434",
        [
            {"name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1"},
            {"name": "llama.cpp", "base_url": "http://127.0.0.1:8080/v1"},
            {"name": "remote server", "base_url": "https://models.example.test/v1"},
        ],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert [(runtime.name, runtime.models) for runtime in runtimes] == [
        ("LM Studio", ("lmstudio-model",)),
        ("llama.cpp", ("gguf-model",)),
        ("Ollama", ("qwen:latest",)),
    ]


def test_local_runtime_discovery_accepts_reachable_server_with_no_models():
    from types import SimpleNamespace

    from agent.local_runtimes import discover_local_runtimes

    class FakeClient:
        is_closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"models": []},
            )

    runtimes = discover_local_runtimes(
        "http://localhost:11434",
        [],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert len(runtimes) == 1
    assert runtimes[0].name == "Ollama"
    assert runtimes[0].models == ()


def test_local_runtime_discovery_survives_and_traces_client_close_failure(
    monkeypatch,
):
    from types import SimpleNamespace

    from agent import local_runtimes
    from agent.local_runtimes import discover_local_runtimes

    events = []
    monkeypatch.setattr(
        local_runtimes,
        "log_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    class FakeClient:
        is_closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"models": [{"name": "local-chat"}]},
            )

        def close(self):
            raise OSError("close failed")

    runtimes = discover_local_runtimes(
        "http://localhost:11434",
        [],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert [(runtime.name, runtime.models) for runtime in runtimes] == [
        ("Ollama", ("local-chat",))
    ]
    assert events == [
        (
            "provider.local_discovery_client_close_failed",
            {
                "level": local_runtimes.logging.WARNING,
                "runtime_name": "Ollama",
                "error_type": "OSError",
                "exc_info": True,
            },
        )
    ]


def test_local_runtime_discovery_rejects_non_loopback_urls():
    from agent.local_runtimes import discover_local_runtimes

    def no_request(**_kwargs):
        raise AssertionError("Non-loopback endpoints must never be contacted.")

    assert (
        discover_local_runtimes(
            "https://remote-ollama.example.test",
            [{"name": "custom", "base_url": "http://192.168.1.10:1234/v1"}],
            client_factory=no_request,
        )
        == []
    )


def test_local_capability_probe_degrades_on_a_plain_exception():
    """A backend error must not escape: `ollama` errors derive from Exception."""
    from types import SimpleNamespace

    from agent.providers import inspect_local_model_capabilities

    class BackendError(Exception):
        """Stands in for ollama.ResponseError, which is not an OSError."""

    class FakeClient:
        def show(self, _model):
            raise BackendError("backend exploded")

    printed = []
    result = inspect_local_model_capabilities(
        "qwen:latest",
        base_url="http://127.0.0.1:11434",
        client_factory=lambda **_kwargs: FakeClient(),
        client_kwargs=lambda: {},
        capability_cache={},
        max_cache_entries=8,
        console=SimpleNamespace(print=lambda *args, **kwargs: printed.append(args)),
        close_client=lambda _client: None,
    )

    assert all(value is None for value in result.values())
    assert printed, "the user must still be told the probe failed"


def test_online_capabilities_treat_a_raising_profile_as_unknown():
    from agent.providers import online_capabilities

    class ExplodingProfile:
        @property
        def profile(self):
            raise RuntimeError("profile exploded")

    unknown = online_capabilities(ExplodingProfile(), True)

    assert all(value is None for value in unknown.values())


def _fake_runtime_client(responses):
    from types import SimpleNamespace

    class FakeClient:
        is_closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: responses[url],
            )

    return lambda **_kwargs: FakeClient()


def test_discovery_lists_every_configured_ollama_server():
    from agent.local_runtimes import discover_local_runtimes

    responses = {
        "http://127.0.0.1:11434/api/tags": {"models": [{"name": "first:latest"}]},
        "http://127.0.0.1:11435/api/tags": {"models": [{"name": "second:latest"}]},
    }

    runtimes = discover_local_runtimes(
        ["http://127.0.0.1:11434", "http://127.0.0.1:11435"],
        [],
        client_factory=_fake_runtime_client(responses),
    )

    assert [(runtime.name, runtime.models) for runtime in runtimes] == [
        ("Ollama 1", ("first:latest",)),
        ("Ollama 2", ("second:latest",)),
    ]


def test_embedding_models_are_filtered_out_for_openai_compatible_servers():
    from agent.local_runtimes import discover_local_runtimes

    responses = {
        "http://127.0.0.1:11434/api/tags": {"models": []},
        "http://127.0.0.1:1234/v1/models": {
            "data": [{"id": "qwen3-8b"}, {"id": "text-embedding-nomic"}]
        },
    }

    runtimes = discover_local_runtimes(
        "http://127.0.0.1:11434",
        [{"name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1"}],
        client_factory=_fake_runtime_client(responses),
    )

    assert [runtime.models for runtime in runtimes] == [("qwen3-8b",), ()]


def test_non_loopback_endpoint_is_used_only_when_remote_hosts_are_allowed():
    from agent.local_runtimes import discover_local_runtimes

    responses = {
        "http://127.0.0.1:11434/api/tags": {"models": []},
        "https://models.example.test/v1/models": {"data": [{"id": "remote"}]},
    }

    rejected = discover_local_runtimes(
        "http://127.0.0.1:11434",
        [{"name": "remote", "base_url": "https://models.example.test/v1"}],
        client_factory=_fake_runtime_client(responses),
    )
    accepted = discover_local_runtimes(
        "http://127.0.0.1:11434",
        [{"name": "remote", "base_url": "https://models.example.test/v1"}],
        client_factory=_fake_runtime_client(responses),
        allow_remote=True,
    )

    assert [runtime.name for runtime in rejected] == ["Ollama"]
    assert [(runtime.name, runtime.models) for runtime in accepted] == [
        ("remote", ("remote",)),
        ("Ollama", ()),
    ]


def test_fetch_local_chat_models_merges_every_reachable_runtime(monkeypatch):
    from types import SimpleNamespace

    from agent import model_selection

    monkeypatch.setattr(
        model_selection,
        "_discover_available_local_runtimes",
        lambda: [
            SimpleNamespace(name="Ollama", models=("qwen3:8b", "shared:latest")),
            SimpleNamespace(name="LM Studio", models=("shared:latest", "llama:3")),
        ],
    )

    assert model_selection.fetch_local_chat_models() == [
        "qwen3:8b",
        "shared:latest",
        "llama:3",
    ]
