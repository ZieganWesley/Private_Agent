"""Manual UI self-check trigger point.

This module exists only to exercise the terminal UI in isolation, without
starting the agent. It is intentionally separate so it can be deleted (or the
``--ui-check`` flag removed) before the code is marked for production.

REMOVE BEFORE PRODUCTION.
"""

from __future__ import annotations

import io
import os
from typing import Any

from rich.console import Console

from .chat_ui import ChatTranscript
from .console import build_console
from .ribbon import ribbon_text, status_ribbon
from .session_ui import SessionUI
from .spinner import StatusIndicator
from .state import StatusState, context_bar, context_state_line
from .theme import ACCESSIBLE_ENV, build_theme


def _render(renderable: Any, width: int) -> str:
    """Render one renderable to plain text at ``width``, without a terminal."""
    buffer = io.StringIO()
    Console(
        theme=build_theme(),
        file=buffer,
        width=width,
        no_color=True,
        force_terminal=False,
    ).print(renderable)
    return buffer.getvalue()


def run_ui_check(console: Any = None) -> int:
    """Exercise the UI components and report any rendering failure.

    Returns 0 on success, 1 on failure.
    """
    target = console if console is not None else build_console()
    failures: list[str] = []

    def record(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    def message_levels() -> None:
        for level in ("info", "success", "warning", "error"):
            getattr(target, level)(f"{level} message")

    def ribbon() -> None:
        state = StatusState(
            model="test-model",
            context_tokens=3,
            context_maximum=10,
            permission_mode="auto",
            output_tokens=128,
            background_processes=1,
        )
        target.print(status_ribbon(state))
        target.print(context_bar(3, 10))
        target.print(context_state_line({"tokens": 3, "max": 10}, 128))

    def accessible() -> None:
        os.environ[ACCESSIBLE_ENV] = "1"
        state = StatusState(
            model="m", context_tokens=3, context_maximum=10, permission_mode="auto"
        )
        target.print(status_ribbon(state, accessible=True))
        indicator = StatusIndicator("Waiting", phase="thinking", console=target)
        rendered = indicator._render()
        assert "(thinking)" in rendered, rendered

    def ribbon_spans() -> None:
        state = StatusState(
            model="test-model",
            context_tokens=9,
            context_maximum=10,
            permission_mode="full",
        )
        styled = ribbon_text(state)
        assert styled.spans, "ribbon rendered without any styled span"
        assert styled.plain == status_ribbon(state)
        assert "\x1b[" not in _render(styled, 80)

    def help_tree_widths() -> None:
        from agent.interactive import SLASH_COMMANDS, help_renderable

        for width in (40, 80, 120):
            rendered = _render(help_renderable(), width)
            assert "\x1b[" not in rendered, f"markup leaked at width {width}"
            overlong = [line for line in rendered.splitlines() if len(line) > width]
            assert not overlong, f"width {width} produced {overlong[:1]}"
            missing = [command for command in SLASH_COMMANDS if command not in rendered]
            assert not missing, f"width {width} dropped {missing[:3]}"

    def tool_request_panel() -> None:
        from agent.tool_display import _tool_request_renderable

        rendered = _render(
            _tool_request_renderable(
                {
                    "name": "read_file",
                    "args": {"path": "src/app.py", "api_key": "secret-value"},
                }
            ),
            80,
        )
        assert "[Tool Request] read_file" in rendered
        assert "api_key" in rendered
        assert "[REDACTED]" in rendered
        assert "secret-value" not in rendered

    def failure_panel_render() -> None:
        from .failure_panel import model_failure_panel

        rendered = _render(
            model_failure_panel(
                "The request did not complete.",
                reference_id="deadbeef",
                model_name="test-model",
                error_type="RuntimeError",
                hints=("`/retry` resends the previous request.",),
            ),
            80,
        )
        for expected in ("deadbeef", "test-model", "RuntimeError", "/retry"):
            assert expected in rendered, expected

    def transcript() -> None:
        chat = ChatTranscript(console=target)
        chat.add_message("user", "hello", echo=True)
        chat.add_message("assistant", "hi there", markdown=False, echo=True)
        chat.stream_chunk("assistant", "streamed", end="", markdown=False)

    def turn() -> None:
        ui = SessionUI(console=target)
        ui.bind(model="m", permission="auto", output_tokens=10)
        ui.set_context(2, 10)
        report = ui.turn_started("hi")
        ui.turn_finished(report)

    def status_bar() -> None:
        ui = SessionUI(console=target)
        ui.bind(model="m", permission="auto", output_tokens=10)
        ui.set_context(2, 10)
        ui._footer.set_busy("Processing", "processing")
        ui._footer.clear_busy()

    for name, fn in (
        ("message_levels", message_levels),
        ("ribbon", ribbon),
        ("ribbon_spans", ribbon_spans),
        ("help_tree", help_tree_widths),
        ("tool_request", tool_request_panel),
        ("failure_panel", failure_panel_render),
        ("accessible", accessible),
        ("transcript", transcript),
        ("turn", turn),
        ("status_bar", status_bar),
    ):
        record(name, fn)

    if failures:
        target.print("[error][UI check] FAILURES:[/error]")
        for failure in failures:
            target.print(f"  - {failure}")
        return 1
    target.print(
        "[success][UI check] All UI components rendered successfully.[/success]"
    )
    return 0


__all__ = ["run_ui_check"]
