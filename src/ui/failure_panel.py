"""The panel shown when a model call fails.

Failures used to be one red line saying "Failed (reference x). Retry or use
``switch``": no tool name, no error type and no statement that the session is
still usable. The panel puts the identifier, the model and the error type in a
small grid, then says plainly what still works and what to do next.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .console import resolve_style
from .markup import sanitize_terminal_text


def model_failure_panel(
    message: str,
    *,
    reference_id: str,
    model_name: str | None = None,
    error_type: str | None = None,
    hints: Sequence[str] = (),
    title: str = "Model request failed",
) -> Panel:
    """Build the failure panel.

    ``message`` is safe for the user: exception text is never included, only
    its type, so endpoint details and credentials cannot leak.
    """
    model_name = sanitize_terminal_text(model_name) if model_name else None
    error_type = sanitize_terminal_text(error_type) if error_type else None
    details = Table.grid(padding=(0, 1))
    details.add_column(style=resolve_style("muted"), justify="right", no_wrap=True)
    details.add_column(style=resolve_style("value"), overflow="fold")
    details.add_row("reference", sanitize_terminal_text(reference_id))
    if model_name:
        details.add_row("model", model_name)
    if error_type:
        details.add_row("error", error_type)

    blocks: list = [
        details,
        Text(sanitize_terminal_text(message), style=resolve_style("value")),
    ]
    for hint in hints:
        blocks.append(Text(sanitize_terminal_text(hint), style=resolve_style("muted")))

    return Panel(
        Group(*blocks),
        title=Text(
            f"\u2717 {sanitize_terminal_text(title)}", style=resolve_style("error")
        ),
        border_style=resolve_style("border.error"),
        padding=(0, 1),
    )


__all__ = ["model_failure_panel"]
