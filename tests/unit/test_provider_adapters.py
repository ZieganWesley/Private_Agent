"""Unit coverage for the OpenAI-compatible adapter layer."""

import pytest

from providers import OpenAICompatibleAdapter, ProviderAdapter


def test_adapter_satisfies_the_protocol():
    assert isinstance(OpenAICompatibleAdapter(), ProviderAdapter)


def test_list_models_reads_discovered_runtime_models():
    adapter = OpenAICompatibleAdapter()

    class Runtime:
        models = ("qwen3:8b", "nomic-embed-text")

    assert adapter.list_models(Runtime()) == ("qwen3:8b", "nomic-embed-text")
    assert adapter.list_models(None) == ()


def test_capabilities_degrade_without_raising(monkeypatch):
    from agent import model_capabilities

    def exploding(*_args, **_kwargs):
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(
        model_capabilities, "detect_openai_compatible_capabilities", exploding
    )

    capabilities = OpenAICompatibleAdapter().capabilities(
        "qwen3:8b",
        base_url="http://127.0.0.1:8080/v1",
        client_factory=object(),
        timeout=0.1,
    )

    assert capabilities == {
        "tools": None,
        "vision": None,
        "thinking": None,
        "context_window": None,
    }


def test_capabilities_use_the_shared_openai_compatible_probe(monkeypatch):
    from agent import model_capabilities

    seen = {}

    def fake_detect(model, *, base_url, client_factory, timeout):
        seen.update(
            model=model,
            base_url=base_url,
            client_factory=client_factory,
            timeout=timeout,
        )
        return {
            "tools": True,
            "vision": False,
            "thinking": None,
            "context_window": 8192,
        }

    monkeypatch.setattr(
        model_capabilities, "detect_openai_compatible_capabilities", fake_detect
    )
    sender = object()

    result = OpenAICompatibleAdapter().capabilities(
        "m", base_url="http://127.0.0.1:8080/v1", client_factory=sender, timeout=1.0
    )

    assert result["context_window"] == 8192
    assert seen == {
        "model": "m",
        "base_url": "http://127.0.0.1:8080/v1",
        "client_factory": sender,
        "timeout": 1.0,
    }


def test_embeddings_degrade_when_the_optional_package_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_openai":
            raise ImportError("langchain_openai is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert (
        OpenAICompatibleAdapter().embeddings(
            "nomic-embed-text", base_url="http://127.0.0.1:8080/v1"
        )
        is None
    )


def test_chat_model_surfaces_a_missing_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_openai":
            raise ImportError("langchain_openai is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        OpenAICompatibleAdapter().chat_model(
            "qwen3:8b", base_url="http://127.0.0.1:8080/v1"
        )
