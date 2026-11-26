"""O4: embedding provider, model, and endpoint resolution."""

from types import SimpleNamespace

from rag.embedding_resolver import resolve_embedding_provider


def _runtime(name, kind, base_url, models=(), embeddings=()):
    return SimpleNamespace(
        name=name,
        kind=kind,
        base_url=base_url,
        models=tuple(models),
        embedding_models=tuple(embeddings),
    )


def test_none_and_openai_compatible_are_unchanged():
    runtimes = [_runtime("ollama", "ollama", "http://127.0.0.1:11434", (), ("nomic",))]

    assert resolve_embedding_provider(runtimes, "none", "m", "u").provider == "none"
    choice = resolve_embedding_provider(runtimes, "openai-compatible", "text-embed", "http://a/v1")
    assert (choice.provider, choice.model, choice.base_url) == (
        "openai-compatible",
        "text-embed",
        "http://a/v1",
    )


def test_explicit_ollama_uses_that_runtimes_model_and_base_url():
    runtimes = [
        _runtime("ollama", "ollama", "http://127.0.0.1:11434", ("llama",), ("nomic-embed",)),
        _runtime("lmstudio", "openai-compatible", "http://127.0.0.1:1234/v1", (), ("bge",)),
    ]

    choice = resolve_embedding_provider(
        runtimes, "ollama", "configured", "http://127.0.0.1:1234/v1"
    )

    assert choice == type(choice)(
        "ollama", "nomic-embed", "http://127.0.0.1:11434"
    )


def test_explicit_ollama_falls_back_to_openai_compatible_like_before():
    runtimes = [
        _runtime(
            "lmstudio", "openai-compatible", "http://127.0.0.1:1234/v1", (), ("bge",)
        )
    ]

    choice = resolve_embedding_provider(runtimes, "ollama", "nomic", "http://x/v1")

    assert (choice.provider, choice.model, choice.base_url) == (
        "openai-compatible",
        "bge",
        "http://127.0.0.1:1234/v1",
    )


def test_ollama_runtime_without_embedding_models_keeps_configured_model():
    runtimes = [_runtime("ollama", "ollama", "http://127.0.0.1:11434")]

    choice = resolve_embedding_provider(runtimes, "ollama", "nomic", "http://x/v1")

    assert (choice.provider, choice.model) == ("ollama", "nomic")


def test_auto_prefers_the_configured_model_when_advertised():
    runtimes = [
        _runtime("ollama", "ollama", "http://127.0.0.1:11434", (), ("nomic",)),
        _runtime("lmstudio", "openai-compatible", "http://127.0.0.1:1234/v1", (), ("bge",)),
    ]

    choice = resolve_embedding_provider(runtimes, "auto", "bge", "http://x/v1")

    assert (choice.provider, choice.model, choice.base_url) == (
        "openai-compatible",
        "bge",
        "http://127.0.0.1:1234/v1",
    )


def test_auto_prefers_ollama_then_openai_then_none():
    ollama = _runtime("o", "ollama", "http://o:11434", (), ("nomic",))
    openai = _runtime("l", "openai-compatible", "http://l:1234/v1", (), ("bge",))

    assert resolve_embedding_provider([openai, ollama], "auto").provider == "ollama"
    assert resolve_embedding_provider([openai], "auto").provider == "openai-compatible"
    assert resolve_embedding_provider([], "auto").provider == "none"


def test_auto_ignores_chat_models_and_unknown_provider_values():
    runtimes = [
        _runtime("lmstudio", "openai-compatible", "http://l:1234/v1", ("chat-only",))
    ]

    assert resolve_embedding_provider(runtimes, "auto").provider == "none"
    assert resolve_embedding_provider(runtimes, "bogus").provider == "none"
