"""Behavioural tests for the extracted streaming/invocation helpers."""

import time
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from private_agent.agent import streaming


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
