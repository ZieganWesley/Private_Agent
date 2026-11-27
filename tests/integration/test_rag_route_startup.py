"""Startup route selection for advertised embedding and rerank models."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import agent.rag_startup as rag_startup
from agent.rag_startup import initialize_rag_session
from providers.selection import STATE_ENV, load_embedding_choice, load_reranker_choice
from rag.rerank import HttpReranker


class _MenuConsole:
    """Rendered Rich output plus scripted numbered-menu answers."""

    def __init__(self, *answers: str) -> None:
        self.answers = iter(answers)
        self.prompts: list[str] = []
        self.output = StringIO()
        self.console = Console(file=self.output, width=120, no_color=True)

    def print(self, *args, **kwargs) -> None:
        self.console.print(*args, **kwargs)

    def input(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        self.console.print(prompt, end="")
        return next(self.answers)

    def status(self, *args, **kwargs):
        return self.console.status(*args, **kwargs)

    @property
    def text(self) -> str:
        return self.output.getvalue()


def _runtime(name: str, port: int, embedding: str, reranker: str):
    return SimpleNamespace(
        name=name,
        base_url=f"http://127.0.0.1:{port}",
        embedding_models=(embedding,),
        records=(SimpleNamespace(model_id=reranker, role="reranker"),),
    )


def _runtimes():
    return (
        _runtime("endpoint-a", 1111, "embed-a", "rank-a"),
        _runtime("endpoint-b", 2222, "embed-b", "rank-b"),
    )


async def _startup(*, runtimes, console, interactive, captured, docs, index):
    state = await initialize_rag_session(
        docs_input=str(docs),
        local_runtimes=runtimes,
        console=console,
        interactive=interactive,
        input_fn=console.input,
        rag_index_path=str(index),
        app_config={},
        initialize_knowledge_base_fn=lambda path, **kwargs: captured.append(kwargs),
        knowledge_base_signature_fn=lambda path: "signature",
    )
    if state[3] is not None:
        await state[3]
    return state


@pytest.mark.asyncio
async def test_startup_menu_uses_and_remembers_both_selected_routes(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "provider-state.json"))
    monkeypatch.setattr(rag_startup, "EMBEDDING_PROVIDER", "auto")
    docs = tmp_path / "docs"
    docs.mkdir()
    first_calls: list[dict] = []
    console = _MenuConsole("2", "2")

    await _startup(
        runtimes=_runtimes(),
        console=console,
        interactive=True,
        captured=first_calls,
        docs=docs,
        index=tmp_path / "index",
    )

    assert "Multiple embedding models are available" in console.text
    assert "Multiple reranker models are available" in console.text
    assert "endpoint-a · embed-a" in console.text
    assert "endpoint-b · embed-b" in console.text
    assert "endpoint-a · rank-a" in console.text
    assert "endpoint-b · rank-b" in console.text
    assert len(console.prompts) == 2
    assert len(first_calls) == 1
    first = first_calls[0]
    assert first["embedding_provider"] == "endpoint-b"
    assert first["embedding_model"] == "embed-b"
    assert first["embedding_base_url"] == "http://127.0.0.1:2222/v1"
    reranker = first["reranker"]
    assert isinstance(reranker, HttpReranker)
    assert reranker.model == "rank-b"
    assert reranker.base_url == "http://127.0.0.1:2222/v1"
    assert load_embedding_choice() == {
        "provider": "endpoint-b",
        "model": "embed-b",
        "base_url": "http://127.0.0.1:2222/v1",
    }
    assert load_reranker_choice() == {
        "provider": "endpoint-b",
        "model": "rank-b",
        "base_url": "http://127.0.0.1:2222/v1",
    }

    second_calls: list[dict] = []
    remembered_console = _MenuConsole()
    await _startup(
        runtimes=_runtimes(),
        console=remembered_console,
        interactive=False,
        captured=second_calls,
        docs=docs,
        index=tmp_path / "index",
    )

    assert remembered_console.prompts == []
    assert len(second_calls) == 1
    remembered = second_calls[0]
    assert remembered["embedding_model"] == "embed-b"
    assert remembered["embedding_base_url"] == "http://127.0.0.1:2222/v1"
    assert remembered["reranker"].model == "rank-b"
    assert remembered["reranker"].base_url == "http://127.0.0.1:2222/v1"


@pytest.mark.asyncio
async def test_startup_without_advertised_models_disables_both_capabilities(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "provider-state.json"))
    monkeypatch.setattr(rag_startup, "EMBEDDING_PROVIDER", "auto")
    docs = tmp_path / "docs"
    docs.mkdir()
    calls: list[dict] = []
    console = _MenuConsole()

    await _startup(
        runtimes=(),
        console=console,
        interactive=True,
        captured=calls,
        docs=docs,
        index=tmp_path / "index",
    )

    assert "semantic search is disabled" in console.text
    assert len(calls) == 1
    assert calls[0]["embedding_provider"] == "none"
    assert calls[0]["embedding_model"] == ""
    assert calls[0]["embedding_base_url"] == ""
    assert calls[0]["reranker"] is None
    assert load_embedding_choice() == {
        "provider": "none",
        "model": "disabled",
        "base_url": "",
    }
    assert load_reranker_choice() == {
        "provider": "none",
        "model": "disabled",
        "base_url": "",
    }
