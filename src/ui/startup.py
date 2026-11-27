"""Startup framing: section rules, key/value summaries and the welcome panel.

Startup output used to be a run of ``[Label] text | text | text`` lines. Long
workspace paths made them wrap into an unreadable block, and interpolated
values were interpreted as rich markup. These renderables keep the same words
while giving each section a rule and each field a label/value split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .console import resolve_style


def startup_rule(title: str) -> Rule:
    """Return a section separator that names the block it opens."""
    return Rule(
        Text(f" {title} ", style=resolve_style("muted")),
        style=resolve_style("border"),
    )


def field_text(label: str, value: Any) -> Text:
    """Return one ``label: value`` line as text (never parsed as markup).

    Long values fold rather than being ellipsized: a cropped workspace path in
    a startup summary is worse than a wrapped one.
    """
    text = Text(overflow="fold")
    text.append(f"{label}: ", style=resolve_style("key"))
    text.append(str(value), style=resolve_style("value"))
    return text


def key_value_columns(
    pairs: Sequence[tuple[str, Any]],
    *,
    padding: tuple[int, int] = (0, 2),
) -> Columns:
    """Lay key/value pairs out in as many columns as the width allows."""
    return Columns(
        [field_text(label, value) for label, value in pairs],
        padding=padding,
        expand=False,
    )


def startup_section(title: str, pairs: Sequence[tuple[str, Any]]) -> Group:
    """A named rule followed by its key/value summary."""
    return Group(startup_rule(title), key_value_columns(pairs))


def figlet_banner(
    title: str = "Private Agent",
    *,
    subtitle: str = "Local Inference & Tool Runtime",
    font: str = "slant",
) -> Panel:
    """Return the startup banner: ASCII-art title and a short subtitle."""
    import pyfiglet

    art = pyfiglet.figlet_format(title, font=font).rstrip("\n")
    heading = Text(art, style=resolve_style("accent_alt"))
    caption = Text(subtitle, style="dim italic")
    return Panel(
        Align.center(Text.assemble(heading, "\n", caption)),
        border_style=resolve_style("border"),
        expand=False,
    )


def welcome_panel(
    model: str,
    workspace: Any,
    *,
    title: str = "Welcome",
    heading: str = "Private Agent",
) -> Panel:
    """Return the welcome panel: a heading plus the two fields that matter."""
    body = Group(
        Text(heading, style=resolve_style("accent")),
        key_value_columns((("Model", model), ("Workspace", workspace))),
    )
    return Panel(body, title=Text(title), border_style=resolve_style("accent"))


__all__ = [
    "figlet_banner",
    "field_text",
    "key_value_columns",
    "startup_rule",
    "startup_section",
    "welcome_panel",
]
