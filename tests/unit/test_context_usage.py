"""C4: real requested/allocated/used context accounting."""

import logging

from agent import context_telemetry
from agent.context_engine import ContextEngine, start_turn_allocation
from agent.context_telemetry import (
    build_context_telemetry,
    format_context_report,
    log_context_allocation,
    print_context_usage,
)

SOURCES = (
    "system",
    "tool_definitions",
    "skills",
    "plan",
    "files",
    "summary",
    "episodic",
    "history",
    "rag",
    "tool_results",
    "user",
)


def _engine():
    return ContextEngine(enabled=True, profile="chat")


def test_requested_allocated_and_used_are_reported_per_source():
    engine = _engine()
    allocation = start_turn_allocation(engine, 1000, "question")
    big = "word " * 4000
    engine.fill_sources(allocation, {"rag": big, "user": "question"})

    telemetry = allocation.telemetry
    assert telemetry["requested"]["rag"] > telemetry["budgets"]["rag"]
    assert telemetry["used"]["rag"] <= telemetry["budgets"]["rag"]
    assert telemetry["requested"]["user"] == telemetry["used"]["user"]
    assert telemetry["sum_requested"] >= telemetry["sum_used"]
    assert telemetry["overflow"]["rag"] > 0
    assert "rag" in telemetry["truncated"]


def test_overflow_is_measured_before_trimming():
    telemetry = build_context_telemetry(
        profile="chat",
        total_tokens=100,
        budgets={"rag": 10},
        sources={"rag": "short"},
        requested={"rag": 50},
    )

    assert telemetry.used["rag"] < 10
    assert telemetry.overflow["rag"] == 40
    assert telemetry.truncated == ["rag"]


def test_context_report_shows_all_three_counts_and_no_content():
    telemetry = build_context_telemetry(
        profile="chat",
        total_tokens=100,
        budgets={"rag": 10, "user": 5},
        sources={"rag": "trimmed text", "user": "secret user prompt"},
        requested={"rag": 40, "user": 3},
    )

    report = format_context_report(telemetry)

    assert "requested=40 allocated=10 used=" in report
    assert "requested=3 allocated=5 used=" in report
    assert "secret user prompt" not in report
    assert "trimmed text" not in report


def test_runtime_report_uses_the_turn_allocation():
    engine = _engine()
    allocation = start_turn_allocation(engine, 500, "question")
    engine.fill_sources(allocation, {"rag": "word " * 500})

    report = context_telemetry.build_runtime_context_report(
        500, [], [], "prompt text", "question", allocation=allocation
    )

    assert report["requested"]["rag"] == allocation.telemetry["requested"]["rag"]
    assert report["sum_budgets"] == allocation.telemetry["sum_budgets"]
    assert report["profile"] == "chat"
    assert "prompt text" not in repr(report)


def test_usage_is_readable_through_the_print_surface():
    printed = []
    telemetry = build_context_telemetry(
        profile="research",
        total_tokens=100,
        budgets={"rag": 10},
        sources={"rag": "text"},
        requested={"rag": 25},
    )

    print_context_usage(telemetry, printed.append)

    assert len(printed) == 1
    assert "Context usage (research)" in printed[0]
    assert "requested=25 allocated=10" in printed[0]


def test_allocation_log_is_redacted(caplog):
    with caplog.at_level(logging.DEBUG):
        log_context_allocation(
            {
                "profile": "chat",
                "total_tokens": 100,
                "budgets": {"rag": 40},
                "requested": {"rag": 90},
                "used": {"rag": 40},
                "truncated": ["rag"],
                "secret": "SECRET DOCUMENT TEXT",
            },
            {"prefix_hash": "abc", "qualifies": True},
        )

    record = next(r for r in caplog.records if getattr(r, "event", "") == "context.allocation")
    fields = record.event_fields
    assert fields["profile"] == "chat"
    assert fields["requested"] == {"rag": 90}
    assert fields["used"] == {"rag": 40}
    assert fields["cache"]["qualifies"] is True
    assert "SECRET DOCUMENT TEXT" not in repr(record.__dict__)
    assert "secret" not in fields


def test_untrimmed_sources_report_zero_overflow():
    engine = _engine()
    allocation = start_turn_allocation(engine, 4000, "question")
    engine.fill_sources(allocation, {"user": "question"})

    assert allocation.telemetry["overflow"]["user"] == 0
    assert allocation.telemetry["truncated"] == []
