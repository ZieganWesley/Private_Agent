"""Selection and persistence of embedding and reranker routes."""

from __future__ import annotations

from agent.rag_routes import (
    choose_embedding_route,
    choose_rerank_route,
    choose_route,
)
from providers.selection import load_embedding_choice, load_reranker_choice
from rag.embedding_resolver import ModelRoute


class _Console:
    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.lines: list[str] = []
        self.prompts: list[str] = []

    def print(self, message: str) -> None:
        self.lines.append(str(message))

    def input(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        return self.answer


def _route(model: str, base_url: str = "http://127.0.0.1:1111/v1", label="local"):
    return ModelRoute("embedding", label, model, base_url)


def test_a_single_route_is_used_without_a_prompt():
    console = _Console()
    route = _route("only")

    assert (
        choose_route((route,), role="embedding", console=console, interactive=True)
        == route
    )
    assert console.prompts == []


def test_multiple_routes_prompt_for_a_choice():
    console = _Console(answer="2")
    routes = (_route("first"), _route("second"))

    assert (
        choose_route(routes, role="embedding", console=console, interactive=True)
        == routes[1]
    )
    assert "Multiple embedding models are available" in "\n".join(console.lines)
    assert console.prompts


def test_an_invalid_choice_uses_the_remembered_default():
    console = _Console(answer="not-a-number")
    routes = (_route("first"), _route("second"))
    remembered = {"model": "second", "base_url": "http://127.0.0.1:1111/v1"}

    assert (
        choose_route(
            routes,
            role="embedding",
            console=console,
            interactive=True,
            remembered=remembered,
        )
        == routes[1]
    )


def test_non_interactive_sessions_use_the_remembered_or_first_route():
    routes = (_route("first"), _route("second"))

    assert (
        choose_route(routes, role="embedding", console=_Console(), interactive=False)
        == routes[0]
    )
    assert (
        choose_route(
            routes,
            role="embedding",
            console=_Console(),
            interactive=False,
            remembered={"model": "second", "base_url": "http://127.0.0.1:1111/v1"},
        )
        == routes[1]
    )


def test_absent_routes_disable_the_capability():
    assert (
        choose_route((), role="embedding", console=_Console(), interactive=True) is None
    )


def test_a_disabled_embedding_route_is_remembered(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))

    assert choose_embedding_route((), console=_Console(), interactive=True) is None
    assert load_embedding_choice()["provider"] == "none"


def test_embedding_and_rerank_choices_are_remembered(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))
    console = _Console()

    embedding = choose_embedding_route(
        (_route("embed", label="endpoint-a"),), console=console, interactive=True
    )
    rerank = choose_rerank_route(
        (ModelRoute("rerank", "endpoint-b", "rank", "http://127.0.0.1:2222/v1"),),
        console=console,
        interactive=True,
    )

    assert embedding is not None and rerank is not None
    assert load_embedding_choice() == {
        "provider": "endpoint-a",
        "model": "embed",
        "base_url": "http://127.0.0.1:1111/v1",
    }
    assert load_reranker_choice() == {
        "provider": "endpoint-b",
        "model": "rank",
        "base_url": "http://127.0.0.1:2222/v1",
    }
