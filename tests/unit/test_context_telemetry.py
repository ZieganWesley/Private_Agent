from agent.context_telemetry import (
    build_context_telemetry,
    configured_context_telemetry,
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


def test_configured_plan_uses_the_whole_prompt_budget():
    plan = configured_context_telemetry()

    assert plan is not None
    assert plan["sum_budgets"] > 0
    assert plan["sum_budgets"] <= plan["total_tokens"]
    assert plan["sum_used"] == 0
    assert {"system", "history", "user"} <= set(plan["budgets"])


def test_configured_plan_respects_an_explicit_window():
    small = configured_context_telemetry(window=4096)
    large = configured_context_telemetry(window=32768)

    assert small is not None and large is not None
    assert small["total_tokens"] == 4096
    assert small["sum_budgets"] < large["sum_budgets"]


def test_context_telemetry_overflow_tracked():
    budgets = {"rag": 1}
    sources = {"rag": "b" * 40}
    tel = build_context_telemetry(total_tokens=10, budgets=budgets, sources=sources)
    data = tel.to_dict()
    assert data["overflow"]["rag"] > 0
    assert "rag" in data["truncated"]
