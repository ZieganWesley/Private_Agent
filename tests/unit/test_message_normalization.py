"""Strict chat templates accept only one leading system message."""

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import streaming
from agent.messages import LATE_SYSTEM_PREFIX, normalize_system_messages


def test_leading_system_messages_are_merged_in_order():
    messages = normalize_system_messages(
        [
            SystemMessage(content="stable rules"),
            SystemMessage(content="runtime context"),
            HumanMessage(content="hi"),
        ]
    )

    assert len([m for m in messages if isinstance(m, SystemMessage)]) == 1
    assert messages[0].content == "stable rules\n\nruntime context"
    assert isinstance(messages[1], HumanMessage)


def test_late_system_messages_become_labelled_user_messages_in_place():
    messages = normalize_system_messages(
        [
            SystemMessage(content="rules"),
            HumanMessage(content="hi"),
            AIMessage(content="working"),
            SystemMessage(content="stop requesting tools"),
        ]
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "rules"
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == f"{LATE_SYSTEM_PREFIX}\nstop requesting tools"


def test_a_single_leading_system_message_is_unchanged():
    messages = normalize_system_messages(
        [SystemMessage(content="rules"), HumanMessage(content="hi")]
    )

    assert len(messages) == 2
    assert messages[0].content == "rules"


@pytest.mark.asyncio
async def test_invoke_with_status_normalizes_before_the_model_call(monkeypatch):
    captured = {}

    async def fake_budget(model, messages, deadline, **kwargs):
        captured["messages"] = list(messages)
        return AIMessage(content="ok")

    monkeypatch.setattr(streaming, "_supports_async_streaming", lambda model: True)
    monkeypatch.setattr(streaming, "_invoke_with_budget", fake_budget)

    await streaming._invoke_with_status(
        object(),
        [
            SystemMessage(content="stable"),
            SystemMessage(content="runtime"),
            HumanMessage(content="hi"),
            AIMessage(content="working"),
            SystemMessage(content="stop"),
        ],
        time.monotonic() + 5,
        model_name="test-model",
        status_message="Working...",
    )

    sent = captured["messages"]
    assert [m for m in sent if isinstance(m, SystemMessage)] == [sent[0]]
    assert sent[0].content == "stable\n\nruntime"
    assert isinstance(sent[-1], HumanMessage)
    assert sent[-1].content.startswith(LATE_SYSTEM_PREFIX)
