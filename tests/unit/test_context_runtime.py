"""Focused tests for live context telemetry and prefix-cache wiring."""

from langchain_core.messages import HumanMessage, SystemMessage

from agent.context_cache import ContextCachePolicy
from agent.context_runtime import prepare_context_request
from agent.context_telemetry import format_context_report


def _prepare(policy, previous_hash=None, user="hello"):
    return prepare_context_request(
        system_prompts=[SystemMessage(content="system rules " * 40)],
        prompt_history=[HumanMessage(content="earlier turn")],
        user_message=HumanMessage(content=user),
        policy=policy,
        provider="openai-compatible",
        previous_hash=previous_hash,
    )


def test_request_prefix_is_measured_and_ordered():
    policy = ContextCachePolicy(enabled=True, min_prefix_tokens=10)

    messages, meta = _prepare(policy)

    assert [message.__class__.__name__ for message in messages] == [
        "SystemMessage",
        "HumanMessage",
        "HumanMessage",
    ]
    assert meta["cache"]["prefix_tokens"] > 10
    assert meta["cache"]["qualifies"] is True
    assert meta["cache"]["provider"]["markers"]["prompt_cache_key"] == "stable_prefix"


def test_unchanged_prefix_is_not_invalidated():
    policy = ContextCachePolicy(enabled=True, min_prefix_tokens=10)
    _, first = _prepare(policy)

    _, second = _prepare(policy, previous_hash=first["cache"]["prefix_hash"])

    assert second["cache"]["invalidated"] is False
    assert second["telemetry"]["invalidations"] == 0


def test_changed_system_prefix_is_invalidated_and_reported():
    policy = ContextCachePolicy(enabled=True, min_prefix_tokens=10)
    _, first = _prepare(policy)
    messages, second = prepare_context_request(
        system_prompts=[SystemMessage(content="different rules")],
        prompt_history=[],
        user_message=HumanMessage(content="hello"),
        policy=policy,
        provider="openai-compatible",
        previous_hash=first["cache"]["prefix_hash"],
    )

    assert messages
    assert second["cache"]["invalidated"] is True
    assert second["telemetry"]["invalidations"] == 1
    report = {"profile": "chat", "cache": second["cache"]}
    assert "cache prefix:" in format_context_report(report)
    assert "invalidated=True" in format_context_report(report)
