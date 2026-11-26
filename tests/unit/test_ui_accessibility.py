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


def test_light_palette_follows_argument_then_environment(monkeypatch):
    from ui.theme import DARK_STYLES, LIGHT_STYLES, build_theme

    monkeypatch.delenv("PRIVATE_AGENT_LIGHT", raising=False)
    dark = build_theme(light=False)
    light = build_theme(light=True)
    assert dark.styles["accent"].color.name == DARK_STYLES["accent"]
    assert light.styles["accent"].color.name == LIGHT_STYLES["accent"]
    assert dark.styles["value"].color.name == DARK_STYLES["value"]

    monkeypatch.setenv("PRIVATE_AGENT_LIGHT", "1")
    assert build_theme().styles["value"].color.name == LIGHT_STYLES["value"]
    assert build_theme(light=False).styles["value"].color.name == DARK_STYLES["value"]


def test_cli_error_and_warning_tags_resolve_against_the_shared_theme():
    import cli
    from ui.console import console as shared

    assert cli.console is shared
    for level in ("info", "success", "warning", "error"):
        style = shared.get_style(level)
        assert style.color is not None, level
        assert style.color.name


def _ribbon_colours(text):
    return {
        span.style.color.name
        for span in text.spans
        if getattr(span.style, "color", None)
    }


def test_ribbon_text_styles_each_segment_by_meaning():
    from ui.ribbon import ribbon_text
    from ui.theme import DARK_STYLES

    state = StatusState(
        model="m",
        context_tokens=9,
        context_maximum=10,
        permission_mode="auto",
        output_tokens=5,
        background_processes=0,
    )
    text = ribbon_text(state)
    colours = _ribbon_colours(text)
    assert text.plain == status_ribbon(state)
    assert DARK_STYLES["ribbon.model"] in colours
    assert DARK_STYLES["ribbon.context.high"] in colours
    assert DARK_STYLES["ribbon.permission.auto"] in colours


def test_ribbon_text_colours_context_and_permission_levels():
    from ui.ribbon import ribbon_text
    from ui.theme import DARK_STYLES

    low = StatusState(
        model="m",
        context_tokens=1,
        context_maximum=10,
        permission_mode="readonly",
    )
    low_colours = _ribbon_colours(ribbon_text(low))
    assert DARK_STYLES["ribbon.context.ok"] in low_colours
    assert DARK_STYLES["ribbon.permission.readonly"] in low_colours

    mid = StatusState(model="m", context_tokens=6, context_maximum=10)
    assert DARK_STYLES["ribbon.context.warn"] in _ribbon_colours(ribbon_text(mid))


def test_ribbon_text_keeps_the_same_words_without_colour():
    from ui.ribbon import ribbon_text

    console = Console(record=True, no_color=True, width=100)
    state = StatusState(
        model="m", context_tokens=3, context_maximum=10, permission_mode="auto"
    )
    console.print(ribbon_text(state))
    rendered = console.export_text()
    assert rendered.strip() == status_ribbon(state)
    assert "\x1b[" not in rendered
