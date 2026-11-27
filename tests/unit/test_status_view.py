"""The shared status panel used by ``/status`` and ``--doctor``."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from ui.status_view import StatusSection, status_panel, status_table


def _render(renderable, width: int = 80) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=width, no_color=True).print(renderable)
    return buffer.getvalue()


def _panel() -> object:
    return status_panel(
        "Session Status",
        (
            StatusSection(
                "Session",
                (
                    ("Provider kind", "Local"),
                    ("Model", "qwen2.5-coder:7b"),
                    ("Workspace", "/home/example/workspace"),
                ),
                ("Setting", "Value"),
            ),
            StatusSection(
                "Context",
                (("Window", "4096 tokens"), ("Used last turn", "512 tokens (12.5%)")),
                ("Measure", "Value"),
            ),
        ),
    )


def test_panel_carries_every_section_and_value():
    rendered = _render(_panel())
    for expected in (
        "Session Status",
        "Session",
        "Context",
        "Provider kind",
        "qwen2.5-coder:7b",
        "/home/example/workspace",
        "4096 tokens",
        "512 tokens (12.5%)",
    ):
        assert expected in rendered, expected


def test_values_are_literal_text_not_markup():
    rendered = _render(
        status_panel(
            "Session Status",
            (StatusSection("Session", (("Model", "[bold]evil[/bold]"),)),),
        )
    )
    assert "[bold]evil[/bold]" in rendered
    assert "\x1b[" not in rendered


def test_sections_with_headers_render_the_header_row():
    rendered = _render(
        status_panel(
            "Doctor",
            (
                StatusSection(
                    "Dependencies",
                    (("OK", "httpx"),),
                    ("Status", "Item"),
                    header=True,
                    justify_first="left",
                ),
            ),
        )
    )
    assert "Status" in rendered
    assert "Item" in rendered
    assert "httpx" in rendered


def test_panel_fits_the_requested_width():
    for width in (40, 80, 120):
        rendered = _render(_panel(), width)
        overlong = [line for line in rendered.splitlines() if len(line) > width]
        assert not overlong, f"width {width} produced {overlong[:1]}"
        assert "\u2026" not in rendered


def test_session_status_panel_reports_context_and_queue_state():
    from agent.commands.status import session_status_panel

    rendered = _render(
        session_status_panel(
            provider_type="local",
            local_runtime_name="Ollama",
            selected_model="qwen2.5-coder:7b",
            permission_mode="ask",
            workspace="/w",
            rag_description="inactive",
            queued_context_present=True,
            queued_context_files=2,
            queued_context_bytes=1024,
            context_value={
                "total_tokens": 4096,
                "sum_used": 512,
                "usage_percent": 12.5,
                "truncated": ["history"],
            },
        )
    )
    for expected in (
        "Local (Ollama)",
        "4096 tokens",
        "512 tokens (12.5%)",
        "Trimmed",
        "history",
        "2 file(s), 1,024 bytes",
    ):
        assert expected in rendered, expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (None, "not requested yet"),
        ({}, "not requested yet"),
        ({"granted": True}, "granted for this session"),
        ({"denied": True}, "denied for this session"),
    ],
)
def test_session_status_panel_reports_full_mode_network_consent(state, expected):
    from agent.commands.status import session_status_panel

    rendered = _render(
        session_status_panel(
            provider_type="local",
            local_runtime_name="Ollama",
            selected_model="qwen2.5-coder:7b",
            permission_mode="full",
            workspace="/w",
            rag_description="inactive",
            queued_context_present=False,
            queued_context_files=0,
            queued_context_bytes=0,
            context_value={},
            network_consent_state=state,
        )
    )

    assert "Internet access" in rendered
    assert expected in rendered


def test_session_status_panel_omits_network_consent_outside_full_mode():
    from agent.commands.status import session_status_panel

    rendered = _render(
        session_status_panel(
            provider_type="local",
            local_runtime_name="Ollama",
            selected_model="qwen2.5-coder:7b",
            permission_mode="auto",
            workspace="/w",
            rag_description="inactive",
            queued_context_present=False,
            queued_context_files=0,
            queued_context_bytes=0,
            context_value={},
            network_consent_state={"granted": True},
        )
    )

    assert "Internet access" not in rendered


def test_session_status_panel_says_when_nothing_was_measured():
    from agent.commands.status import session_status_panel

    rendered = _render(
        session_status_panel(
            provider_type="online",
            local_runtime_name="",
            selected_model="gpt",
            permission_mode="allow",
            workspace="/w",
            rag_description="inactive",
            queued_context_present=False,
            queued_context_files=0,
            queued_context_bytes=0,
            context_value={},
        )
    )
    assert "Online" in rendered
    assert "not measured yet" in rendered
    assert "none" in rendered


def test_panel_renders_on_a_console_without_the_theme():
    table = status_table(StatusSection("Session", (("Model", "m"),), ("Key", "Value")))
    assert "Model" in _render(table)
