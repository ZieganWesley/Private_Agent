"""Accessibility and terminal-width tests for the CLI UI."""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest
from rich.console import Console

from ui import SessionUI
from ui.ribbon import status_ribbon
from ui.state import StatusState

_UI_CHECK_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ui_check.py"
)


@pytest.fixture
def ui_check():
    """Load the developer UI self-check script from its source path.

    It lives outside ``src/ui`` so it is not importable from an installed
    wheel; the tests exercise the script file itself.
    """
    spec = importlib.util.spec_from_file_location("ui_check", _UI_CHECK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert "Model: m" in text
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
        assert segs[0].startswith("Model: ")
        assert segs[1].startswith("Context: ")
        assert segs[2].startswith("Permission: ")
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
    assert "Mode: local" in bound.segments()
    assert not any(s.startswith("Mode:") for s in StatusState().segments())


def test_session_ui_ribbon_reports_provider_mode():
    console = Console(record=True, width=120)
    ui = SessionUI(console=console)
    ui.bind(model="m", permission="ask", provider_mode="online")
    ui.set_context(1, 4)
    ribbon = ui.ribbon_text(width=120)
    assert "Mode: online" in ribbon
    assert "Permission: ask" in ribbon


def test_ribbon_segments_use_human_labels():
    state = StatusState(
        model="qwen",
        context_tokens=1,
        context_maximum=4,
        permission_mode="ask",
        output_tokens=64,
        background_processes=1,
        provider_mode="local",
    )
    segs = state.segments()
    assert segs == [
        "Model: qwen",
        "Context: 25.0%",
        "Permission: ask",
        "Mode: local",
        "Output: 64",
        "Background: 1",
    ]
    # No raw key=value keys survive in the user-facing ribbon.
    assert not any("=" in segment for segment in segs)


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


def test_model_failure_panel_shows_reference_and_never_the_exception_text():
    import io

    from ui.failure_panel import model_failure_panel

    panel = model_failure_panel(
        "The request did not complete.",
        reference_id="deadbeef",
        model_name="local-model",
        error_type="RuntimeError",
        hints=("`/retry` resends the previous request.",),
    )
    output = io.StringIO()
    Console(file=output, width=100, no_color=True).print(panel)
    rendered = output.getvalue()

    assert "Model request failed" in rendered
    assert "deadbeef" in rendered
    assert "local-model" in rendered
    assert "RuntimeError" in rendered
    assert "/retry" in rendered
    assert "The request did not complete." in rendered


def test_model_failure_panel_neutralises_terminal_sequences():
    import io

    from ui.failure_panel import model_failure_panel

    panel = model_failure_panel(
        "The request did not complete.",
        reference_id="deadbeef",
        model_name="local\x1b[2Jmodel",
        error_type="RuntimeError\x1b]0;pwned\x07",
    )
    output = io.StringIO()
    Console(file=output, width=100, no_color=True).print(panel)
    rendered = output.getvalue()

    assert "\x1b" not in rendered
    assert "localmodel" in rendered
    assert "RuntimeError" in rendered


def test_ui_check_covers_every_component_and_passes(ui_check, capsys):
    assert ui_check.run_ui_check() == 0
    assert "All UI components rendered successfully" in capsys.readouterr().out


def test_ui_check_reports_a_component_failure(ui_check, monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(ui_check, "status_ribbon", boom)

    assert ui_check.run_ui_check() == 1
    assert "render exploded" in capsys.readouterr().out


def test_prompt_text_is_plain_without_colour(monkeypatch):
    from ui.prompt import prompt_text

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("PRIVATE_AGENT_ACCESSIBLE", raising=False)

    assert prompt_text("User: ") == "User: "


def test_prompt_text_is_plain_in_accessible_mode(monkeypatch):
    from ui.prompt import prompt_text

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("PRIVATE_AGENT_ACCESSIBLE", "1")

    assert prompt_text("User: ") == "User: "


def test_prompt_text_follows_the_active_palette(monkeypatch):
    from rich.console import Console

    from ui.prompt import prompt_text
    from ui.theme import build_theme

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("PRIVATE_AGENT_ACCESSIBLE", raising=False)
    dark = Console(
        force_terminal=True, color_system="truecolor", theme=build_theme(light=False)
    )
    light = Console(
        force_terminal=True, color_system="truecolor", theme=build_theme(light=True)
    )

    dark_prompt = prompt_text("User: ", console=dark)
    light_prompt = prompt_text("User: ", console=light)

    assert "User: " in dark_prompt.value
    assert "User: " in light_prompt.value
    assert "\x1b[" in dark_prompt.value
    assert dark_prompt.value != light_prompt.value


def test_ui_check_does_not_leak_the_accessible_flag(ui_check, monkeypatch):
    """Probing accessible mode must not change later renders in this process."""
    from ui.theme import ACCESSIBLE_ENV

    monkeypatch.delenv(ACCESSIBLE_ENV, raising=False)
    assert ui_check.run_ui_check() == 0
    assert ACCESSIBLE_ENV not in os.environ


def test_ui_check_restores_the_accessible_flag_even_on_failure(ui_check, monkeypatch):
    from ui.theme import ACCESSIBLE_ENV

    monkeypatch.setenv(ACCESSIBLE_ENV, "1")

    def boom(*_args, **_kwargs):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(ui_check, "status_ribbon", boom)
    assert ui_check.run_ui_check() == 1
    assert os.environ[ACCESSIBLE_ENV] == "1"


def test_ui_check_exercises_every_colour_depth(ui_check, monkeypatch, capsys):
    from rich.console import Console as RealConsole

    depths = []

    class RecordingConsole(RealConsole):
        def __init__(self, *args, **kwargs):
            depths.append(kwargs.get("color_system"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ui_check, "Console", RecordingConsole)

    assert ui_check.run_ui_check() == 0
    capsys.readouterr()
    assert {"truecolor", "256", "standard"} <= set(depths)
    assert None in depths


def test_ui_check_runs_as_a_script(tmp_path):
    import subprocess
    import sys

    import ui

    source_root = pathlib.Path(ui.__file__).resolve().parent.parent
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONPATH": os.pathsep.join(
            [str(source_root), os.environ.get("PYTHONPATH", "")]
        ),
    }

    result = subprocess.run(
        [sys.executable, str(_UI_CHECK_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "All UI components rendered successfully" in result.stdout


def test_ui_check_is_not_importable_from_the_shipped_package():
    import importlib.util

    assert importlib.util.find_spec("ui.check") is None
