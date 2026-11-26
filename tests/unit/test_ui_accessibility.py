"""Accessibility and terminal-width tests for the CLI UI."""

from __future__ import annotations

from rich.console import Console

from ui import SessionUI
from ui.ribbon import status_ribbon
from ui.state import StatusState


def test_no_color_output_has_no_ansi_escapes(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    console = Console(record=True, width=120)
    ui = SessionUI(console=console)
    ui.bind(model="m", permission="auto", output_tokens=10)
    ui.set_context(2, 10)
    report = ui.turn_started("hi")
    ui.turn_finished(report)
    text = console.export_text()
    assert "\x1b[" not in text
    assert "model=m" in text
    assert "[Turn]" in text


def test_narrow_width_keeps_essential_segments():
    for width in (30, 40, 50, 60):
        state = StatusState(
            model="very-long-model-name",
            context_tokens=3,
            context_maximum=10,
            permission_mode="auto",
            output_tokens=2048,
            background_processes=2,
        )
        segs = state.segments(width=width)
        assert len(segs) >= 3
        assert segs[0].startswith("model=")
        assert segs[1].startswith("context=")
        assert segs[2].startswith("permission=")
        # Dropping only removes trailing optional segments.
        assert len(segs) <= 7


def test_accessible_mode_renders_text_gauge(monkeypatch):
    monkeypatch.setenv("PRIVATE_AGENT_ACCESSIBLE", "1")
    state = StatusState(
        model="m", context_tokens=3, context_maximum=10, permission_mode="auto"
    )
    ribbon = status_ribbon(state, accessible=True)
    assert "3/10 (30%)" in ribbon
    assert "\u25b0" not in ribbon


def test_spinner_render_includes_phase_when_accessible(monkeypatch):
    monkeypatch.setenv("PRIVATE_AGENT_ACCESSIBLE", "1")
    from rich.console import Console

    from ui.spinner import StatusIndicator

    console = Console(record=True)
    indicator = StatusIndicator("Waiting", phase="thinking", console=console)
    rendered = indicator._render()
    assert "(thinking)" in rendered


def test_segments_never_drop_below_three():
    state = StatusState(
        model="m", context_tokens=1, context_maximum=2, permission_mode="x"
    )
    for width in (1, 5, 10, 20):
        assert len(state.segments(width=width)) >= 3


def test_reduced_motion_prints_one_static_status_line(monkeypatch):
    monkeypatch.setenv("PRIVATE_AGENT_REDUCED_MOTION", "1")
    from ui.spinner import StatusIndicator, motion_reduced

    assert motion_reduced() is True
    console = Console(record=True, width=80)
    indicator = StatusIndicator("Waiting for llama", phase="loading", console=console)
    indicator.start()
    assert indicator.active is True
    # Updates do not redraw, so nothing moves on screen.
    indicator.update(phase="thinking", message="Thinking")
    indicator.stop()
    text = console.export_text()
    assert "Waiting for llama" in text
    assert "Thinking" not in text
    assert indicator.active is False


def test_reduced_motion_off_by_default(monkeypatch):
    monkeypatch.delenv("PRIVATE_AGENT_REDUCED_MOTION", raising=False)
    from ui.spinner import motion_reduced

    assert motion_reduced() is False


def test_provider_mode_segment_is_shown_when_bound():
    bound = StatusState(
        model="m",
        context_tokens=1,
        context_maximum=2,
        permission_mode="auto",
        provider_mode="local",
    )
    assert "mode=local" in bound.segments()
    assert not any(s.startswith("mode=") for s in StatusState().segments())


def test_session_ui_ribbon_reports_provider_mode():
    console = Console(record=True, width=120)
    ui = SessionUI(console=console)
    ui.bind(model="m", permission="ask", provider_mode="online")
    ui.set_context(1, 4)
    ribbon = ui.ribbon_text(width=120)
    assert "mode=online" in ribbon
    assert "permission=ask" in ribbon
