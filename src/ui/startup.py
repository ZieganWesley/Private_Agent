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
from rich.table import Table
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
    subtitle: str = "READY · Local-first AI assistant with repository intelligence, memory and tools",
    font: str = "slant",
    width: int | None = None,
) -> Panel:
    """Return the startup banner: ASCII-art title and a short subtitle.

    The reference UI falls back to a plain uppercase title when the ASCII art
    would not fit the terminal. Keeping the fallback here avoids a broken,
    wrapped figlet drawing in narrow windows.
    """
    import pyfiglet

    art = pyfiglet.figlet_format(title, font=font).rstrip("\n")
    art_width = max((len(line) for line in art.splitlines()), default=0)
    fits = width is None or art_width + 6 <= width
    heading = Text(
        art if fits else title.upper(),
        style=resolve_style("accent_alt" if fits else "accent"),
    )
    caption = Text(subtitle, style="dim italic")
    return Panel(
        Align.center(Text.assemble(heading, "\n", caption)),
        border_style=resolve_style("border.banner"),
        padding=(0, 2),
        expand=True,
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


# Capability flags rendered as one compact icon per capability: yes, unknown,
# unsupported. The order is fixed so two sessions compare at a glance.
CAPABILITY_ICONS: tuple[tuple[str, str, str, str], ...] = (
    ("tools", "🔧", "🛠️", "🚫"),
    ("function_calls", "⚡", "❓", "➖"),
    ("structured_output", "📐", "❔", "➖"),
    ("thinking", "🧠", "💭", "➖"),
    ("vision", "👁️", "🙈", "🙈"),
    ("audio", "🎙️", "🔇", "🔇"),
)


def capability_icons(capabilities: dict[str, Any] | None) -> str:
    """Render capability flags as the reference UI's compact icon row."""
    flags = capabilities or {}
    icons: list[str] = []
    for key, yes_icon, unknown_icon, no_icon in CAPABILITY_ICONS:
        value = str(flags.get(key, "no")).strip().lower()
        if value == "yes":
            icons.append(yes_icon)
        elif value == "unknown":
            icons.append(unknown_icon)
        else:
            icons.append(no_icon)
    return "".join(icons)


def _runtime_cell(value: Any) -> Text:
    """Return a runtime value as literal text so markup never eats it."""
    return value if isinstance(value, Text) else Text(str(value))


def _rag_status(rag: Any, rag_active: bool, rag_detail: str) -> Text:
    """Return the reference UI's three-state local-RAG label."""
    label = str(rag).strip().lower()
    if label in {"", "disabled", "none", "off"}:
        return Text("disabled", style=resolve_style("muted"))
    if rag_active:
        return Text("active", style=resolve_style("border.success"))
    detail = f" ({rag_detail})" if rag_detail else ""
    return Text(f"inactive{detail}", style=resolve_style("border.warning"))


def runtime_info_panel(
    *,
    model: str,
    provider: str,
    permission: str,
    isolation: str,
    workspace: Any,
    rag: Any,
    rag_active: bool,
    skills: int,
    capabilities: dict[str, Any] | None = None,
    platform: str = "",
    rag_detail: str = "",
    command_footer: str = "Commands: /help · /mode · /think · /net · /up · /down · switch · exit",
) -> Panel:
    """Return the production startup panel: runtime facts and capability badges.

    This is the reference UI's ``Model & Runtime Info`` panel. Two aligned
    columns keep the session's identity, safety mode and environment readable
    without wrapping into a stack of ``[Label] text | text`` lines.
    """
    caps_text = (
        ", ".join(f"{key}={value}" for key, value in (capabilities or {}).items())
        or "—"
    )
    caps = Text(f"{skills} · ")
    caps.append(caps_text, style=resolve_style("muted"))

    left_rows: list[tuple[str, Any]] = [
        ("Model", model),
        ("Provider", provider),
        ("Permission", permission),
        ("Isolation", isolation),
    ]
    right_rows: list[tuple[str, Any]] = [
        ("Workspace", str(workspace)),
        ("RAG Dir", str(rag)),
        ("Local RAG", _rag_status(rag, rag_active, rag_detail)),
        ("Skills / Caps", caps),
    ]

    # One four-column grid keeps the reference's two halves side by side.
    # Nested expanding tables measure as full-width and stack instead.
    body = Table.grid(padding=(0, 2), expand=True)
    body.add_column(style=resolve_style("key"), width=12, no_wrap=True)
    body.add_column(style=resolve_style("value"), ratio=1, overflow="fold")
    body.add_column(style=resolve_style("key"), width=13, no_wrap=True)
    body.add_column(style=resolve_style("value"), ratio=1, overflow="fold")
    for index in range(max(len(left_rows), len(right_rows))):
        left = left_rows[index] if index < len(left_rows) else ("", "")
        right = right_rows[index] if index < len(right_rows) else ("", "")
        body.add_row(
            Text(left[0]),
            _runtime_cell(left[1]),
            Text(right[0]),
            _runtime_cell(right[1]),
        )

    return Panel(
        body,
        title=Text("Model & Runtime Info", style=resolve_style("border.runtime")),
        subtitle=Text(
            command_footer,
            style=resolve_style("muted"),
        ),
        border_style=resolve_style("border.runtime"),
        padding=(0, 1),
    )


def ready_header(
    *,
    model: str,
    provider: str,
    permission: str,
    isolation: str,
    workspace: Any,
    rag: Any,
    rag_active: bool,
    skills: int,
    capabilities: dict[str, Any] | None = None,
    platform: str = "",
    rag_detail: str = "",
    title: str = "Private Agent",
    subtitle: str = "READY · Local-first AI assistant with repository intelligence, memory and tools",
    font: str = "slant",
    width: int | None = None,
    command_footer: str = "Commands: /help · /mode · /think · /net · /up · /down · switch · exit",
) -> Group:
    """Return the combined production ready screen.

    The reference UI does not print the banner and runtime facts as separate
    blocks. Keeping them in one renderable also lets the command footer travel
    with the runtime panel instead of being rediscovered from scattered prints.
    """
    return Group(
        figlet_banner(title=title, subtitle=subtitle, font=font, width=width),
        runtime_info_panel(
            model=model,
            provider=provider,
            permission=permission,
            isolation=isolation,
            workspace=workspace,
            rag=rag,
            rag_active=rag_active,
            skills=skills,
            capabilities=capabilities,
            platform=platform,
            rag_detail=rag_detail,
            command_footer=command_footer,
        ),
    )


__all__ = [
    "CAPABILITY_ICONS",
    "capability_icons",
    "figlet_banner",
    "field_text",
    "key_value_columns",
    "runtime_info_panel",
    "startup_rule",
    "startup_section",
    "ready_header",
    "welcome_panel",
]
