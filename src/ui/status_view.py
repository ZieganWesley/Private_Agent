"""One shape for session status and doctor output.

``/status`` used to build an ad-hoc table with a hardcoded ``cyan`` key column
and no frame, while ``--doctor`` built six separate tables with their own
colours. Both now render the same ``Panel`` of titled sections: one border
token, one accent colour, aligned values, and every cell as ``Text`` so a model
name or a workspace path can never be interpreted as markup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .console import resolve_style


@dataclass(frozen=True)
class StatusSection:
    """A titled group of rows under a fixed column layout."""

    title: str
    rows: Sequence[Sequence[Any]] = field(default_factory=tuple)
    columns: Sequence[str] = ()
    header: bool = False
    justify_first: str = "right"


def status_table(section: StatusSection) -> Table:
    """Render one section body: keys right-aligned, values folding."""
    table = Table.grid(padding=(0, 2), expand=False)
    for index, _column in enumerate(section.columns):
        table.add_column(
            justify=section.justify_first if index == 0 else "left",
            style=resolve_style("key" if index == 0 else "value"),
            no_wrap=index == 0,
            overflow="fold",
        )
    if section.header and section.columns:
        table.add_row(
            *[Text(column, style=resolve_style("muted")) for column in section.columns]
        )
    for row in section.rows:
        table.add_row(*[_cell(cell) for cell in row])
    return table


def _cell(value: Any) -> Text:
    if isinstance(value, Text):
        return value
    return Text(str(value), overflow="fold")


def status_panel(
    title: str,
    sections: Sequence[StatusSection],
    *,
    subtitle: str | None = None,
) -> Panel:
    """Render titled sections inside one bordered panel."""
    body: list[Any] = []
    for index, section in enumerate(sections):
        if index:
            body.append(Text(""))
        if section.title:
            body.append(Text(section.title, style=resolve_style("accent")))
        body.append(status_table(section))
    return Panel(
        Group(*body),
        title=Text(title, style=resolve_style("accent")),
        subtitle=Text(subtitle) if subtitle else None,
        border_style=resolve_style("border"),
        padding=(0, 1),
    )


__all__ = ["StatusSection", "status_panel", "status_table"]
