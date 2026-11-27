"""RAG routes are resolved once and reused while still available."""

from __future__ import annotations

from types import SimpleNamespace

from agent.rag_routes import (
    remembered_reranker_model,
    resolve_embedding_route,
    save_reranker_route,
)
from providers.selection import (
    load_embedding_choice,
    save_embedding_choice,
)


class _Console:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, message: str) -> None:
        self.lines.append(str(message))


def _runtime(kind: str, base_url: str, embeddings: tuple[str, ...]):
    return SimpleNamespace(
        kind=kind,
        base_url=base_url,
        embedding_models=embeddings,
    )


def _run(
    *,
    console: _Console,
    provider: str,
    model: str,
    runtimes,
):
    return resolve_embedding_route(
        configured_provider=provider,
        configured_model=model,
        configured_base_url="http://configured/v1",
        runtimes=runtimes,
        console=console,
    )


def test_auto_reuses_the_remembered_embedding_route(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))
    save_embedding_choice(
        provider="openai-compatible",
        model="bge",
        base_url="http://127.0.0.1:1234/v1",
    )
    console = _Console()

    choice = _run(
        console=console,
        provider="auto",
        model="nomic",
        runtimes=[
            _runtime("ollama", "http://127.0.0.1:11434", ("nomic",)),
            _runtime("openai-compatible", "http://127.0.0.1:1234/v1", ("bge",)),
        ],
    )

    assert (choice.provider, choice.model) == ("openai-compatible", "bge")
    assert any("remembered embedding route" in line for line in console.lines)


def test_a_missing_remembered_route_is_resolved_and_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))
    console = _Console()

    choice = _run(
        console=console,
        provider="auto",
        model="nomic",
        runtimes=[_runtime("ollama", "http://127.0.0.1:11434", ("nomic",))],
    )

    assert (choice.provider, choice.model) == ("ollama", "nomic")
    assert load_embedding_choice() == {
        "provider": "ollama",
        "model": "nomic",
        "base_url": "http://127.0.0.1:11434",
    }


def test_a_disabled_embedding_route_is_never_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))

    choice = _run(console=_Console(), provider="none", model="m", runtimes=[])

    assert choice.provider == "none"
    assert load_embedding_choice() is None


def test_the_remembered_reranker_model_is_read_and_written(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))
    assert remembered_reranker_model() is None

    save_reranker_route("bge-reranker", "http://127.0.0.1:9000/v1")

    assert remembered_reranker_model() == "bge-reranker"
