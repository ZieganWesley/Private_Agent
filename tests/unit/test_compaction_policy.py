"""C5: trim deterministically before asking the summarizer."""

from langchain_core.messages import AIMessage, HumanMessage

from agent.compaction import messages_tokens
from agent.context_engine import ContextEngine, plan_turn_compaction


def history(count, text="word " * 50):
    return [
        (HumanMessage if i % 2 == 0 else AIMessage)(content=f"{i} {text}")
        for i in range(count)
    ]


def test_history_below_threshold_needs_no_compaction():
    engine = ContextEngine(enabled=True, profile="chat")
    messages = history(3)

    decision = engine.should_compact(messages, 10, 100_000, 0.8, 20, keep_recent=4)

    assert decision.reason == "below_threshold"
    assert decision.history == messages


def test_trimming_that_fits_skips_the_summarizer():
    engine = ContextEngine(enabled=True, profile="chat")
    messages = history(6)
    trimmed, needs_compact = plan_turn_compaction(
        engine, messages, 0, 200, 0.8, 20, keep_recent=2
    )

    assert needs_compact is False
    assert len(trimmed) < len(messages)
    assert messages_tokens(trimmed) <= 0.8 * 200
    assert trimmed == messages[len(messages) - len(trimmed) :]


def test_message_count_pressure_is_solved_by_trimming():
    engine = ContextEngine(enabled=True, profile="chat")
    messages = history(22, text="tiny")

    decision = engine.should_compact(messages, 0, 100_000, 0.8, 20, keep_recent=2)

    assert decision.reason == "trim_sufficient"
    assert len(decision.history) == 20


def test_compaction_runs_only_when_even_the_floor_still_overflows():
    engine = ContextEngine(enabled=True, profile="chat")
    messages = history(6, text="word " * 500)

    decision = engine.should_compact(messages, 0, 100, 0.8, 20, keep_recent=4)

    assert decision.reason == "compact"
    assert len(decision.history) == 4
    assert decision.post_trim_tokens > 80
    assert (
        plan_turn_compaction(engine, messages, 0, 100, 0.8, 20, keep_recent=4)[1]
        is True
    )


def test_empty_or_disabled_threshold_never_compacts():
    engine = ContextEngine(enabled=True, profile="chat")

    assert engine.should_compact([], 0, 100, 0.8, 20).reason == "below_threshold"
    assert engine.should_compact(history(30), 0, 100, 0, 20).reason == "below_threshold"
