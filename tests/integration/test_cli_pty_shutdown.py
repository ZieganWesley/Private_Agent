"""End-to-end PTY coverage for the CLI's shutdown screen and interactive commands."""

from __future__ import annotations

import sys

import pytest
from pty_cli import ansi_screen, plain_text, run_cli_in_pty

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="PTY tests require a POSIX terminal"
)


def test_cli_pty_erases_the_ribbon_after_goodbye(tmp_path):
    raw, _ = run_cli_in_pty(tmp_path)
    screen = ansi_screen(raw)
    lines = screen.split("\n")

    assert "Permission: full" in plain_text(raw), "the status ribbon never rendered"
    goodbye = [index for index, line in enumerate(lines) if "Goodbye." in line]
    assert goodbye, screen
    last_goodbye = goodbye[-1]

    after = "\n".join(lines[last_goodbye:])
    assert "Model:" not in after
    assert "Context:" not in after
    assert "Permission: full" not in after
    assert all(not line.strip() for line in lines[last_goodbye + 1 :]), screen
    assert last_goodbye + 1 < len(lines)
    assert lines[last_goodbye + 1] == ""


def test_cli_pty_network_consent_waits_and_blocks_the_tool(tmp_path):
    tool_marker = tmp_path / "tool-ran.txt"

    def assert_tool_has_not_run(marker):
        if b"Full permission mode does not include internet access" in marker:
            assert not tool_marker.exists(), "the tool ran before consent was answered"

    raw, _ = run_cli_in_pty(
        tmp_path,
        replies=(
            (b"User:", b"search for the runtime release notes\n"),
            (b"Full permission mode does not include internet access", b"n\n"),
            (b"User:", b"exit\n"),
        ),
        extra_env={
            "PA_PTY_TOOL_CALL": "1",
            "PA_TOOL_MARKER": str(tool_marker),
        },
        before_reply=assert_tool_has_not_run,
    )
    plain = plain_text(raw)
    screen = ansi_screen(raw)

    assert "Full permission mode does not include internet access" in plain, plain
    assert "Full permission mode does not include internet access" in screen, screen
    assert "web_search" in plain
    assert "release notes for the local runtime" in plain
    assert "[y/N]" in plain
    assert "network request declined" in plain, plain
    assert not tool_marker.exists(), "the declined tool must not run"
    assert "Goodbye." in plain, plain


def test_cli_pty_answers_the_doctor_confirmation_prompt(tmp_path):
    raw, _ = run_cli_in_pty(
        tmp_path,
        replies=(
            (b"User:", b"/doctor\n"),
            (b"Is this information correct?", b"n\n"),
            (b"User:", b"exit\n"),
        ),
    )
    plain = plain_text(raw)

    assert "Private Agent Doctor" in plain, plain
    assert "Is this information correct? [y/N]:" in plain, plain
    assert "Manual config changes cannot be performed" in plain, plain
    assert "Goodbye." in plain, plain
