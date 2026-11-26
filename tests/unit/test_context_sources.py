"""Unit tests for context source adapters."""

from agent import context_sources as cs
from agent.context_manager import ContextSource


def test_trim_text_to_tokens_respects_budget():
    text = "word " * 1000
    trimmed = cs.trim_text_to_tokens(text, 10)
    assert cs.trim_text_to_tokens("", 10) == ""
    assert cs.trim_text_to_tokens(text, 0) == ""
    # token_count of trimmed should not exceed budget by much
    from agent.prompts import token_count

    assert token_count(trimmed) <= 10 + 2


def test_adapters_return_context_source():
    src = cs.adapt_history(["hello", "world"], budget_tokens=5)
    assert isinstance(src, ContextSource)
    assert src.name == "history"
    assert src.text  # non-empty trimmed


def test_adapter_respects_budget():
    long_text = "token " * 500
    src = cs.adapt_rag(text=long_text, budget_tokens=8)
    from agent.prompts import token_count

    assert token_count(src.text) <= 8 + 2


def test_build_source_unknown_name():
    src = cs.build_source("custom_source", text="abc", budget_tokens=10)
    assert src.name == "custom_source"
    assert src.text == "abc"


def test_class_adapters():
    adapter = cs.HistorySourceAdapter()
    src = adapter.build(messages=["a", "b"], budget_tokens=4)
    assert src.name == "history"


def test_adapters_do_not_change_output_when_disabled():
    # Adapters are pure functions; calling twice yields same result.
    text = "stable " * 50
    a = cs.adapt_skills(text=text, budget_tokens=6)
    b = cs.adapt_skills(text=text, budget_tokens=6)
    assert a.text == b.text
    assert a.priority == b.priority


def test_source_priorities_and_cacheable():
    src = cs.adapt_system(text="sys", budget_tokens=10)
    assert src.priority >= 90
    assert src.cacheable is True
    rag = cs.adapt_rag(text="doc", budget_tokens=10)
    assert rag.cacheable is False
