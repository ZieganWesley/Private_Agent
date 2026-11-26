from agent.context_telemetry import (
    build_context_telemetry,
    format_context_report,
)
from agent.prompts import token_count as _tc


def test_context_telemetry_sums_correctly():
    budgets = {"system": 10, "rag": 5}
    sources = {"system": "a" * 40, "rag": "b" * 10}
    tel = build_context_telemetry(
        profile="chat", total_tokens=100, budgets=budgets, sources=sources
    )
    data = tel.to_dict()
    assert data["sum_budgets"] == 15
    assert data["sum_used"] == _tc("a" * 40) + _tc("b" * 10)
    assert data["usage_percent"] >= 0
    assert "system" in data["budgets"]
    # no content leaked
    report = format_context_report(data)
    assert "a" * 40 not in report


def test_context_telemetry_overflow_tracked():
    budgets = {"rag": 1}
    sources = {"rag": "b" * 40}
    tel = build_context_telemetry(total_tokens=10, budgets=budgets, sources=sources)
    data = tel.to_dict()
    assert data["overflow"]["rag"] > 0
    assert "rag" in data["truncated"]
