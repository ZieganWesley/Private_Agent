"""Behavioural tests for the extracted streaming/invocation helpers."""

import time
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from agent import streaming


def test_visible_chunk_text_reads_strings_and_text_blocks():
    assert streaming._visible_chunk_text("plain") == "plain"
    assert (
        streaming._visible_chunk_text(
            [
                {"type": "text", "text": "a"},
                "b",
                {"type": "image_url", "image_url": "ignored"},
                {"type": "text", "text": 5},
            ]
        )
        == "ab"
    )
    assert streaming._visible_chunk_text(None) == ""


def test_model_response_is_empty_considers_content_and_tool_calls():
    assert streaming._model_response_is_empty(None)
    assert streaming._model_response_is_empty(AIMessage(content="   "))
    assert not streaming._model_response_is_empty(AIMessage(content="answer"))
    assert not streaming._model_response_is_empty(
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
        )
    )


def test_visible_reasoning_summary_prefers_labeled_summaries_only():
    response = types.SimpleNamespace(
        additional_kwargs={"reasoning": "private chain of thought"},
        response_metadata={"reasoning_summary": "  Compared two options.  "},
    )
    raw_only = types.SimpleNamespace(
        additional_kwargs={"reasoning": "private chain of thought"},
        response_metadata={},
    )

    assert streaming._visible_reasoning_summary(response) == "Compared two options."
    assert streaming._visible_reasoning_summary(raw_only) == ""


def test_streaming_support_requires_async_generator_and_configuration(monkeypatch):
    async def stream(messages):
        yield messages

    model = types.SimpleNamespace(astream=stream)
    monkeypatch.setattr(streaming, "STREAMING_OUTPUT", True)
    assert streaming._supports_async_streaming(model)

    monkeypatch.setattr(streaming, "STREAMING_OUTPUT", False)
    assert not streaming._supports_async_streaming(model)
    assert not streaming._supports_async_streaming(types.SimpleNamespace())


@pytest.mark.asyncio
async def test_invoke_with_budget_uses_single_call_when_streaming_is_disabled(
    monkeypatch,
):
    monkeypatch.setattr(streaming, "STREAMING_OUTPUT", False)
    monkeypatch.setattr(streaming.console, "print", lambda *_a, **_k: None)
    expected = AIMessage(content="complete")
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=expected)
    model.astream = AsyncMock()

    response = await streaming._invoke_with_budget(model, [], time.monotonic() + 5)

    assert response is expected
    model.ainvoke.assert_awaited_once_with([])
    model.astream.assert_not_awaited()


@pytest.mark.asyncio
async def test_invoke_with_budget_retries_an_empty_response(monkeypatch):
    monkeypatch.setattr(streaming, "STREAMING_OUTPUT", False)
    monkeypatch.setattr(streaming.console, "print", lambda *_a, **_k: None)
    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=[AIMessage(content=""), AIMessage(content="recovered")]
    )
    model.astream = AsyncMock()

    response = await streaming._invoke_with_budget(
        model, [], time.monotonic() + 5, model_name="test-model"
    )

    assert response.content == "recovered"
    assert model.ainvoke.await_count == 2


def test_generation_rate_reports_provider_tokens(monkeypatch):
    printed = []
    monkeypatch.setattr(
        streaming.console,
        "print",
        lambda *values, **_kwargs: printed.append(values[0]),
    )
    monkeypatch.setattr(streaming, "VERBOSE_TIMINGS", True)

    streaming._display_generation_rate(
        types.SimpleNamespace(
            content="visible output", usage_metadata={"output_tokens": 10}
        ),
        2,
    )

    assert "10 model-reported output tokens" in printed[0]
    assert "5.0 tokens/s" in printed[0]


def test_generation_rate_skips_empty_output(monkeypatch):
    printed = []
    monkeypatch.setattr(
        streaming.console,
        "print",
        lambda *values, **_kwargs: printed.append(values[0]),
    )

    streaming._display_generation_rate(
        types.SimpleNamespace(content="", usage_metadata={"output_tokens": 0}), 1
    )

    assert printed == []


def test_visible_reasoning_display_respects_configuration(monkeypatch):
    printed = []
    monkeypatch.setattr(
        streaming.console,
        "print",
        lambda *values, **_kwargs: printed.append(values[0]),
    )
    response = types.SimpleNamespace(
        additional_kwargs={},
        response_metadata={"reasoning_summary": "Compared two options."},
    )

    monkeypatch.setattr(streaming, "VISIBLE_REASONING", False)
    streaming._display_visible_reasoning(response)
    assert printed == []

    monkeypatch.setattr(streaming, "VISIBLE_REASONING", True)
    streaming._display_visible_reasoning(response)
    assert printed[0].title.plain == "Model rationale summary"


class FakeIndicator:
    """Record indicator lifecycle calls without starting a real rich Live."""

    def __init__(self):
        self.active = False
        self.phase = streaming.PROCESSING
        self.starts = 0
        self.stops = 0
        self.updates = []

    def start(self):
        self.active = True
        self.starts += 1
        return self

    def update(self, *, phase=None, message=None):
        if phase is not None:
            self.phase = phase
        self.updates.append((phase, message))

    def stop(self):
        if self.active:
            self.stops += 1
        self.active = False


def _patch_streaming(monkeypatch, *, visible_reasoning):
    indicator = FakeIndicator()
    printed = []
    monkeypatch.setattr(streaming, "StatusIndicator", lambda *_a, **_k: indicator)
    monkeypatch.setattr(streaming, "VISIBLE_REASONING", visible_reasoning)
    monkeypatch.setattr(streaming, "STREAMING_OUTPUT", True)
    monkeypatch.setattr(
        streaming.console,
        "print",
        lambda *values, **kwargs: printed.append((values, kwargs)),
    )
    return indicator, printed


@pytest.mark.asyncio
async def test_streaming_reasoning_is_rendered_as_dimmed_text(monkeypatch):
    indicator, printed = _patch_streaming(monkeypatch, visible_reasoning=True)

    class ReasoningModel:
        async def astream(self, _messages):
            yield AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": "weigh options"}
            )
            yield AIMessageChunk(content="final answer")

    response = await streaming._invoke_with_budget(
        ReasoningModel(), [], time.monotonic() + 5, model_name="m"
    )

    assert response.content == "final answer"
    assert any(
        values == ("weigh options",) and kwargs.get("style") == "dim"
        for values, kwargs in printed
    )
    assert any("Thinking (m)" in str(values[0]) for values, _kwargs in printed)
    assert indicator.starts == 1
    assert indicator.stops == 1


@pytest.mark.asyncio
async def test_hidden_reasoning_switches_the_spinner_to_thinking(monkeypatch):
    indicator, printed = _patch_streaming(monkeypatch, visible_reasoning=False)

    class ReasoningModel:
        async def astream(self, _messages):
            yield AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": "weigh options"}
            )
            yield AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": "pick one"}
            )
            yield AIMessageChunk(content="final answer")

    response = await streaming._invoke_with_budget(
        ReasoningModel(), [], time.monotonic() + 5, model_name="m"
    )

    assert response.content == "final answer"
    assert indicator.updates == [(streaming.THINKING, "m is thinking")]
    assert not any(kwargs.get("style") == "dim" for _values, kwargs in printed)
    assert not any("weigh options" in str(values) for values, _kwargs in printed)


@pytest.mark.asyncio
async def test_context_usage_reaches_the_streaming_spinner(monkeypatch):
    created = []

    class RecordingIndicator(FakeIndicator):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created.append((args, kwargs))

    monkeypatch.setattr(streaming, "StatusIndicator", RecordingIndicator)
    monkeypatch.setattr(streaming, "STREAMING_OUTPUT", True)
    monkeypatch.setattr(streaming, "VISIBLE_REASONING", False)
    monkeypatch.setattr(streaming.console, "print", lambda *_a, **_k: None)

    class SimpleModel:
        async def astream(self, _messages):
            yield AIMessageChunk(content="hello")

    await streaming._invoke_with_budget(
        SimpleModel(),
        [],
        time.monotonic() + 5,
        model_name="m",
        context_usage=(3000, 12000),
    )

    assert created[0][1]["context_usage"] == (3000, 12000)


@pytest.mark.asyncio
async def test_streamed_text_is_printed_only_after_the_spinner_stops(monkeypatch):
    indicator, printed = _patch_streaming(monkeypatch, visible_reasoning=False)
    created_live = []

    class FakeLive:
        def __init__(self, renderable, *, console=None, **kwargs):
            self.renderable = renderable
            self.started_while_spinner_active = indicator.active
            created_live.append(self)

        def start(self):
            return None

        def update(self, renderable):
            self.renderable = renderable

        def stop(self):
            return None

    from ui import markdown_view

    monkeypatch.setattr(markdown_view, "Live", FakeLive)

    class SimpleModel:
        async def astream(self, _messages):
            yield AIMessageChunk(content="hello")

    await streaming._invoke_with_budget(
        SimpleModel(), [], time.monotonic() + 5, model_name="m"
    )

    assert created_live
    assert all(not live.started_while_spinner_active for live in created_live)
    assert indicator.stops == 1
