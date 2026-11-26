"""Context budget bars and the accessible text fallback."""

from __future__ import annotations

import io

from rich.console import Console

from agent.context_telemetry import build_context_telemetry, print_context_usage
from ui.context_view import bars_supported, budget_bar, context_report_panel
from ui.theme import ACCESSIBLE_ENV


def _telemetry():
    return build_context_telemetry(
        profile="chat",
        total_tokens=8192,
        budgets={"system": 800, "history": 400},
        sources={"system": "s " * 100, "history": "h " * 900},
        requested={"system": 900, "history": 900},
    )


def _render(renderable, width: int = 80) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=width, no_color=True, color_system=None).print(
        renderable
    )
    return buffer.getvalue()


def test_report_lists_each_source_with_used_and_budget():
    rendered = _render(context_report_panel(_telemetry()))
    assert "Context usage (chat)" in rendered
    assert "history" in rendered
    assert "system" in rendered
    assert "400" in rendered
    assert "800" in rendered
    assert "12.2% of window" in rendered


def test_overflow_is_marked_not_silently_clipped():
    rendered = _render(context_report_panel(_telemetry()))
    assert "901 overflow" in rendered
    assert "trimmed: system, history" in rendered


def test_bars_fit_the_requested_width():
    for width in (60, 80, 120):
        rendered = _render(context_report_panel(_telemetry()), width)
        overlong = [line for line in rendered.splitlines() if len(line) > width]
        assert not overlong, f"width {width} produced {overlong[:1]}"


def test_budget_bar_clamps_between_zero_and_total():
    full = budget_bar(50, 10)
    assert full.total == 10
    assert full.completed == 10
    empty = budget_bar(-5, 10)
    assert empty.completed == 0
    assert budget_bar(1, 0).total == 1


def test_bars_are_only_used_on_a_colour_console(monkeypatch):
    monkeypatch.delenv(ACCESSIBLE_ENV, raising=False)
    assert bars_supported(Console(file=io.StringIO(), color_system="truecolor"))
    assert not bars_supported(Console(file=io.StringIO(), color_system=None))
    assert not bars_supported(
        Console(file=io.StringIO(), color_system="truecolor", no_color=True)
    )
    assert not bars_supported(None)
    monkeypatch.setenv(ACCESSIBLE_ENV, "1")
    assert not bars_supported(Console(file=io.StringIO(), color_system="truecolor"))


def test_print_context_usage_switches_between_bars_and_text(monkeypatch):
    monkeypatch.delenv(ACCESSIBLE_ENV, raising=False)
    telemetry = _telemetry()
    colour_console = Console(file=io.StringIO(), color_system="truecolor")
    printed = []
    print_context_usage(telemetry, printed.append, console=colour_console)
    assert len(printed) == 1
    assert not isinstance(printed[0], str)

    plain = []
    print_context_usage(telemetry, plain.append)
    assert isinstance(plain[0], str)
    assert "requested=900 allocated=800" in plain[0]
