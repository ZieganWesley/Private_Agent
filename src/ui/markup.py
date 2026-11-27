"""Escape untrusted text before it is interpolated into rich markup.

Chrome lines like ``f"[info][Plan][/info] {text}"`` run the interpolated value
through rich's markup parser, so a model reply or a file path containing
``[bold]`` changed the styling and a value like ``[not-a-style]`` raised
``MarkupError`` and lost the line. Every interpolation of user, model or
filesystem text into markup goes through :func:`literal`.

Markup is not the only way untrusted text rewrites the screen. Escaping
brackets leaves ANSI and OSC sequences untouched, and the terminal runs those
as commands: clear the screen, retitle the window, overwrite the scrollback or
read and set the clipboard. Every print of model, file or exception text goes
through :func:`sanitize_terminal_text` first.
"""

from __future__ import annotations

import re
from typing import Any

from rich.markup import escape

# One pass over every escape a terminal would act on. The specific sequences
# come first so a complete sequence is consumed whole; the trailing character
# classes then guarantee that no escape byte survives a truncated or malformed
# sequence either.
_CONTROL_SEQUENCE = re.compile(
    "|".join(
        (
            # OSC (window title, clipboard, hyperlinks), BEL or ST terminated.
            # An unterminated one stops at the end of its line so the rest of
            # the reply stays readable.
            r"\x1b\][^\x07\x1b\n]*(?:\x07|\x1b\\)?",
            # DCS, SOS, PM and APC strings, ST terminated.
            r"\x1b[P^_X][^\x1b\n]*(?:\x1b\\)?",
            # CSI: parameters, intermediates, then a final byte.
            r"\x1b\[[0-?]*[ -/]*[@-~]",
            # Any other ESC sequence: charset selection, single-character escapes.
            r"\x1b[ -/]*[0-~]",
            # The same three families in their 8-bit C1 form.
            r"\x9b[0-?]*[ -/]*[@-~]",
            r"\x9d[^\x07\x1b\n]*(?:\x07|\x1b\\)?",
            r"\x90[^\x1b\n]*(?:\x1b\\)?",
            # Bare ESC and every remaining C0 control. Newline and tab are
            # content, not control, and stay.
            r"[\x00-\x08\x0b-\x1f\x7f]",
            # Remaining single-byte C1 controls.
            r"[\x80-\x9f]",
        )
    )
)

# Cheap pre-check: the common reply contains no control bytes at all.
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


def sanitize_terminal_text(value: Any) -> str:
    """Return ``value`` with terminal control sequences removed.

    Model output, file contents and exception text are untrusted, and a
    terminal executes escape sequences instead of showing them. Newlines and
    tabs survive, so a reply keeps its layout.
    """
    text = str(value)
    if _CONTROL_CHARACTER.search(text) is None:
        return text
    return _CONTROL_SEQUENCE.sub("", text)


def literal(value: Any) -> str:
    """Return ``value`` as markup-safe, terminal-safe text."""
    return escape(sanitize_terminal_text(value))


def plan_line(plan_text: Any) -> str:
    """Chrome line for the model's plan of a multi-step turn."""
    return f"[info][Plan][/info] {literal(plan_text)}"


def citation_line(citation: Any) -> str:
    """Chrome line for one retrieved source."""
    return f"  - {literal(citation)}"


def verification_line(verification: Any) -> str:
    """Chrome line for the code-task verification summary."""
    return f"[info][Code Task Verification][/info] {literal(verification)}"


def assistant_header(model_name: Any) -> str:
    """Chrome header printed before an assistant reply."""
    return f"\n[bold success]AI ({literal(model_name)}):[/bold success] "


def reasoning_header(model_name: Any) -> str:
    """Chrome header printed before visible reasoning."""
    return f"\n[dim]Thinking ({literal(model_name)}):[/dim] "


__all__ = [
    "assistant_header",
    "citation_line",
    "literal",
    "plan_line",
    "reasoning_header",
    "sanitize_terminal_text",
    "verification_line",
]
