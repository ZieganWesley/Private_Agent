"""Untrusted text stays literal wherever it is interpolated into markup."""

from __future__ import annotations

import io

from rich.console import Console

from ui.markup import (
    assistant_header,
    citation_line,
    literal,
    plan_line,
    reasoning_header,
    sanitize_terminal_text,
    verification_line,
)

HOSTILE = "see [bold]b[/bold] and [/] plus [not-a-style]"
PLAIN = "see [bold]b[/bold] and [/] plus [not-a-style]"


def _render(*values) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, no_color=True)
    for value in values:
        console.print(value, end="")
    return buffer.getvalue()


def test_literal_escapes_brackets_before_they_reach_the_parser():
    assert literal(HOSTILE).count("\\[") == 4
    assert literal(None) == "None"
    assert literal(42) == "42"


def test_unescaped_content_would_be_parsed_as_style():
    # Guard for the regression this sweep fixed: an unescaped tag is consumed,
    # so the user never sees the model's literal text.
    assert _render("[bold]b[/bold]").strip() == "b"
    assert PLAIN in _render(plan_line(HOSTILE))


def test_plan_line_shows_model_brackets_without_eating_them():
    rendered = _render(plan_line(HOSTILE))
    assert rendered.startswith("[Plan] ")
    assert PLAIN in rendered


def test_citation_line_and_verification_line_escape_content():
    assert PLAIN in _render(citation_line(HOSTILE))
    rendered = _render(verification_line(HOSTILE))
    assert "[Code Task Verification]" in rendered
    assert PLAIN in rendered


def test_assistant_and_reasoning_headers_escape_the_model_name():
    rendered = _render(assistant_header("[bold]evil"), reasoning_header("[/]"))
    assert "AI ([bold]evil):" in rendered
    assert "Thinking ([/]):" in rendered


def test_approval_diff_escapes_file_content():
    from agent.permissions import _approval_diff

    diff = _approval_diff(
        "notes.py",
        {
            "old_text": "value = '[bold]old[/bold]'",
            "new_text": "value = '[not-a-style]new'",
        },
    )

    assert diff is not None
    assert "\\[bold]old\\[/bold]" in diff
    assert "\\[not-a-style]new" in diff
    # The diff chrome keeps working through the escaped content.
    assert _render(diff).count("value =") == 2


def test_escaped_content_does_not_leak_style_into_the_next_line():
    colour = io.StringIO()
    console = Console(
        file=colour, width=200, color_system="truecolor", force_terminal=True
    )
    console.print(plan_line(HOSTILE))
    console.print("after " + literal("plain"))
    tail = colour.getvalue().rsplit("after", 1)[1]
    assert "\x1b[1m" not in tail


def test_terminal_control_sequences_are_neutralised():
    hostile = (
        "before "
        "\x1b[2J"  # clear the screen
        "\x1b]0;pwned\x07"  # retitle the window (BEL terminated)
        "\x1b]52;c;aGFjaw==\x1b\\"  # clipboard read (ST terminated)
        "\x1bPq#0;2;0;0;0\x1b\\"  # device control string
        "\x9b1m"  # 8-bit CSI
        "\x00\x08\x7f"  # other control bytes
        " after"
    )

    cleaned = sanitize_terminal_text(hostile)

    assert cleaned == "before  after"
    assert not any(
        char == "\x1b" or 0x80 <= ord(char) <= 0x9F or ord(char) < 0x20
        for char in cleaned
    )


def test_newlines_and_tabs_are_content_and_survive():
    layout = "first line\n\tindented\n\nlast line"

    assert sanitize_terminal_text(layout) == layout
    # A carriage return is a control character, not layout.
    assert sanitize_terminal_text("progress\rdone") == "progressdone"
    assert sanitize_terminal_text("plain reply") == "plain reply"


def test_markup_is_still_escaped_after_sanitising():
    escaped = literal("\x1b[2J[bold]x[/bold]")

    assert escaped == r"\[bold]x\[/bold]"
    assert "\x1b" not in escaped


def test_unterminated_osc_loses_only_its_own_line():
    assert sanitize_terminal_text("\x1b]0;title\nnext line") == "\nnext line"


def test_hostile_sequence_never_reaches_the_console():
    rendered = _render(literal("model output \x1b[2J\x1b]0;pwned\x07 with \x1b[31mred"))

    assert "\x1b" not in rendered
    assert "model output  with red" in rendered
