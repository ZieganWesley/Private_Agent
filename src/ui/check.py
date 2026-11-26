"""Manual UI self-check trigger point.

This module exists only to exercise the terminal UI in isolation, without
starting the agent. It is intentionally separate so it can be deleted (or the
``--ui-check`` flag removed) before the code is marked for production.

REMOVE BEFORE PRODUCTION.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console

from .chat_ui import ChatTranscript
from .console import build_console
from .ribbon import ribbon_text, status_ribbon
from .session_ui import SessionUI
from .spinner import StatusIndicator
from .state import StatusState, context_bar, context_state_line
from .theme import ACCESSIBLE_ENV, build_theme


@contextmanager
def _temporary_env(**values: str) -> Iterator[None]:
    """Set environment variables for one check and restore them afterwards.

    ``run_ui_check`` runs in the same process as the caller (and the test
    suite), so leaving ``PRIVATE_AGENT_ACCESSIBLE`` behind would silently
    change every later render.
    """
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        with _temporary_env(**{ACCESSIBLE_ENV: "1"}):
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

    def startup_summary() -> None:
        from .startup import startup_section, welcome_panel

        pairs = (
            ("Config file", "/home/example/private_config.json"),
            ("RAG documents", "not configured"),
            ("RAG index", "/home/example/.private_agent/rag_index"),
            ("Skills", "/home/example/skills"),
        )
        for width in (40, 80, 120):
            rendered = _render(startup_section("Configuration", pairs), width)
            assert "Configuration" in rendered, f"missing rule title at {width}"
            for label, _value in pairs:
                assert label in rendered, f"width {width} dropped {label}"
            assert "\u2026" not in rendered, f"width {width} truncated a value"
            overlong = [line for line in rendered.splitlines() if len(line) > width]
            assert not overlong, f"width {width} produced {overlong[:1]}"
            welcome = _render(
                welcome_panel("test-model", "/home/example/workspace"), width
            )
            assert "test-model" in welcome, f"welcome lost the model at {width}"
            assert "Workspace" in welcome, f"welcome lost the workspace at {width}"

    def status_panel_render() -> None:
        from .status_view import StatusSection, status_panel

        panel = status_panel(
            "Session Status",
            (
                StatusSection(
                    "Session",
                    (
                        ("Provider kind", "Local"),
                        ("Model", "test-model"),
                        ("Workspace", "/home/example/workspace"),
                    ),
                    ("Setting", "Value"),
                ),
                StatusSection(
                    "Context",
                    (("Window", "4096 tokens"), ("Used last turn", "512 tokens")),
                    ("Measure", "Value"),
                ),
            ),
        )
        for width in (40, 80, 120):
            rendered = _render(panel, width)
            for expected in (
                "Session Status",
                "Provider kind",
                "test-model",
                "Context",
            ):
                assert expected in rendered, f"width {width} lost {expected}"
            overlong = [line for line in rendered.splitlines() if len(line) > width]
            assert not overlong, f"width {width} produced {overlong[:1]}"

    def context_bars() -> None:
        from agent.context_telemetry import build_context_telemetry

        from .context_view import context_report_panel

        telemetry = build_context_telemetry(
            profile="chat",
            total_tokens=4096,
            budgets={"system": 400, "history": 200},
            sources={"system": "s " * 50, "history": "h " * 400},
            requested={"system": 500, "history": 400},
        )
        for width in (60, 80, 120):
            rendered = _render(context_report_panel(telemetry), width)
            assert "Context usage" in rendered, f"no title at width {width}"
            for expected in ("history", "system", "total", "overflow"):
                assert expected in rendered, f"width {width} lost {expected}"
            overlong = [line for line in rendered.splitlines() if len(line) > width]
            assert not overlong, f"width {width} produced {overlong[:1]}"

    def tool_result_footer() -> None:
        from agent.tool_display import TOOL_RESULT_PREVIEW_CHARS, _tool_result_panel

        long_body = "line of output\n" * (TOOL_RESULT_PREVIEW_CHARS // 5)
        for width in (60, 80, 120):
            rendered = _render(_tool_result_panel("read_file", long_body), width)
            assert "Tool output: read_file" in rendered
            assert "lines \u00b7 preview of" in rendered
            assert "/expand" in rendered
            overlong = [line for line in rendered.splitlines() if len(line) > width]
            assert not overlong, f"width {width} produced {overlong[:1]}"
        short = _render(_tool_result_panel("read_file", "short"), 80)
        assert "/expand" not in short

    def markdown_stream() -> None:
        from .markdown_view import MarkdownStream, print_markdown

        buffer = io.StringIO()
        plain = Console(file=buffer, width=80, no_color=True)
        print_markdown(plain, "```python\ndef answer():\n    return 42\n```")
        rendered = buffer.getvalue()
        assert "def answer()" in rendered, "complete reply was not rendered"
        assert "```" not in rendered, "fence leaked into a complete reply"

        stream_buffer = io.StringIO()
        stream = MarkdownStream(
            Console(file=stream_buffer, width=80, no_color=True), refresh_per_second=8
        )
        for chunk in ("```pyth", "on\ndef ans", "wer():\n    return 42\n", "```"):
            stream.update(chunk)
        assert stream.renders == 1, f"burst parsed {stream.renders} times"
        stream.stop()
        streamed = stream_buffer.getvalue()
        assert "def answer()" in streamed, "streamed reply lost text"
        assert "```" not in streamed, "fence leaked from the streamed reply"

    def markup_escapes() -> None:
        from .markup import assistant_header, plan_line

        hostile = "see [bold]b[/bold] and [/] plus [not-a-style]"
        rendered = _render(plan_line(hostile), 80)
        assert "[Plan]" in rendered, "plan chrome label was lost"
        assert "[not-a-style]" in rendered, "hostile markup was consumed"
        header = _render(assistant_header(hostile), 120)
        assert "AI (see [bold]b[/bold]" in header, "model name was escaped wrongly"

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
        ("startup_summary", startup_summary),
        ("status_panel", status_panel_render),
        ("context_bars", context_bars),
        ("tool_result_footer", tool_result_footer),
        ("markdown_stream", markdown_stream),
        ("markup_escapes", markup_escapes),
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
