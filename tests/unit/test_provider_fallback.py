"""Unit coverage for mid-turn provider fallback (I1o)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from private_agent.agent.provider_fallback import (
    _recover_on_another_provider,
    next_local_provider,
)


def _runtime(name, models=("m",), kind="openai-compatible"):
    return SimpleNamespace(
        name=name, models=models, kind=kind, base_url="http://127.0.0.1:1/v1"
    )


def test_next_local_provider_skips_the_excluded_runtime():
    runtimes = [_runtime("Ollama"), _runtime("LM Studio")]

    candidate = next_local_provider(exclude_name="Ollama", discover=lambda: runtimes)

    assert candidate is not None
    assert candidate[0].name == "LM Studio"


def test_next_local_provider_returns_none_when_nothing_else_runs():
    assert (
        next_local_provider(
            exclude_name="Ollama", discover=lambda: [_runtime("Ollama")]
        )
        is None
    )


def test_next_local_provider_returns_none_when_discovery_fails():
    def exploding():
        raise RuntimeError("port scan failed")

    assert next_local_provider(exclude_name="Ollama", discover=exploding) is None


def test_recover_on_another_provider_returns_none_when_there_is_no_candidate():
    console = SimpleNamespace(print=lambda *_args, **_kwargs: None)

    result = __import__("asyncio").run(
        _recover_on_another_provider(
            selected_local_runtime=_runtime("Ollama"),
            tools_list=[],
            messages=[],
            deadline=1.0,
            thinking_enabled=False,
            thinking_effort="medium",
            usage={},
            console=console,
            memory=None,
            session_id="s",
            interactive=True,
            input_fn=lambda _prompt: "y",
            discover=lambda: [_runtime("Ollama")],
        )
    )

    assert result is None


def test_recover_on_another_provider_returns_none_when_declined():
    console = SimpleNamespace(print=lambda *_args, **_kwargs: None)

    result = __import__("asyncio").run(
        _recover_on_another_provider(
            selected_local_runtime=_runtime("Ollama"),
            tools_list=[],
            messages=[],
            deadline=1.0,
            thinking_enabled=False,
            thinking_effort="medium",
            usage={},
            console=console,
            memory=None,
            session_id="s",
            interactive=True,
            input_fn=lambda _prompt: "no",
            discover=lambda: [_runtime("Ollama"), _runtime("LM Studio")],
        )
    )

    assert result is None


def test_recover_on_another_provider_retries_and_persists_on_accept(monkeypatch):
    from private_agent.agent import provider_fallback

    console_messages = []
    console = SimpleNamespace(
        print=lambda message, **_kwargs: console_messages.append(str(message))
    )
    recovered_runtime = _runtime("LM Studio")

    monkeypatch.setattr(
        provider_fallback,
        "_make_selected_local_chat_model",
        lambda runtime, model, **kwargs: SimpleNamespace(runtime=runtime, model=model),
    )
    monkeypatch.setattr(provider_fallback, "_bind_tools", lambda llm, tools: llm)
    monkeypatch.setattr(
        provider_fallback,
        "_invoke_with_status",
        AsyncMock(return_value="response"),
    )
    monkeypatch.setattr(
        provider_fallback, "persist_provider_choice", lambda **kwargs: None
    )

    result = __import__("asyncio").run(
        _recover_on_another_provider(
            selected_local_runtime=_runtime("Ollama"),
            tools_list=["tool"],
            messages=["hi"],
            deadline=1.0,
            thinking_enabled=True,
            thinking_effort="high",
            usage={"prompt_tokens": 1},
            console=console,
            memory=None,
            session_id="s",
            interactive=True,
            input_fn=lambda _prompt: "yes",
            discover=lambda: [_runtime("Ollama"), recovered_runtime],
        )
    )

    assert result is not None
    assert result[2] == "m"
    assert result[5] is recovered_runtime
    assert any("Provider switched" in message for message in console_messages)
