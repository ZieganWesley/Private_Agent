"""Escape untrusted text before it is interpolated into rich markup.

Chrome lines like ``f"[info][Plan][/info] {text}"`` run the interpolated value
through rich's markup parser, so a model reply or a file path containing
``[bold]`` changed the styling and a value like ``[not-a-style]`` raised
``MarkupError`` and lost the line. Every interpolation of user, model or
filesystem text into markup goes through :func:`literal`.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape


def literal(value: Any) -> str:
    """Return ``value`` as markup-safe text with its own brackets intact."""
    return escape(str(value))


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
    "verification_line",
]
