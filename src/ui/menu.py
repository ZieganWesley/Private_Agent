"""Rich menu rendering and numbered selection.

The runtime has several menus that fall back to numbered text when the
arrow-key selector is unavailable. This module gives them one panel layout and
one selection rule, so every menu looks the same and maps numbers to values
consistently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .console import resolve_style


@dataclass(frozen=True)
class MenuOption:
    """One menu row: the value returned, its label, and an optional hint."""

    value: str
    label: str
    hint: str = ""


def _entries(options: Sequence[Any]) -> tuple[MenuOption, ...]:
    entries: list[MenuOption] = []
    for option in options:
        if isinstance(option, MenuOption):
            entries.append(option)
        elif len(option) == 2:
            entries.append(MenuOption(str(option[0]), str(option[1])))
        else:
            entries.append(MenuOption(str(option[0]), str(option[1]), str(option[2])))
    return tuple(entries)


def menu_panel(
    title: str,
    options: Sequence[Any],
    *,
    subtitle: str | None = None,
) -> Panel:
    """Render ``options`` as a compact, numbered Rich panel."""
    entries = _entries(options)
    lines: list[Text] = []
    for index, entry in enumerate(entries, 1):
        line = Text()
        line.append(f"[{index}]", style=resolve_style("key"))
        line.append(" ")
        line.append(entry.label, style=resolve_style("value"))
        if entry.hint:
            line.append("  ")
            line.append(entry.hint, style=resolve_style("muted"))
        lines.append(line)
    body = Text("\n").join(lines) if lines else Text("No options.")
    content: Any = body
    if subtitle:
        content = Text.assemble((subtitle, resolve_style("muted")), "\n\n", body)
    return Panel(
        content,
        title=Text(title, style=resolve_style("accent")),
        border_style=resolve_style("border"),
        expand=False,
    )


def select_menu(
    title: str,
    options: Sequence[Any],
    *,
    console: Any,
    input_fn: Any = None,
    default: str | None = None,
    prompt: str = "Choose",
    subtitle: str | None = None,
    allow_custom: bool = False,
    allow_cancel: bool = False,
) -> str | None:
    """Show a Rich menu and return the chosen option value.

    ``options`` accepts ``MenuOption`` or ``(value, label[, hint])`` tuples.
    A number selects by position; an exact value also works for text menus.
    ``allow_custom`` returns an unrecognised answer unchanged, which lets a
    model menu accept a typed model id. Invalid answers return ``default``.
    """
    entries = _entries(options)
    if not entries:
        return None
    console.print(menu_panel(title, options, subtitle=subtitle))
    choose = input_fn if input_fn is not None else console.input
    values = [entry.value for entry in entries]
    default_display = str(values.index(default) + 1) if default in values else default
    message = f"{prompt} [1-{len(entries)}]"
    if default_display:
        message += f" (Default: {escape(str(default_display))})"
    try:
        answer = str(choose(f"[info]{message}: [/info]")).strip()
    except EOFError, KeyboardInterrupt:
        return None
    if not answer:
        return default
    if allow_cancel and answer.lower() in {"q", "quit", "cancel", "exit", "esc"}:
        return None
    if answer in values:
        return answer
    if answer.isdigit() and 1 <= int(answer) <= len(entries):
        return entries[int(answer) - 1].value
    if allow_custom:
        return answer
    console.print(f"[warning][Menu][/warning] Invalid choice: {answer}")
    return default


__all__ = ["MenuOption", "menu_panel", "select_menu"]
