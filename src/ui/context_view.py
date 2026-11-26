"""Context budget rendering: one bar per source, plus the total.

``/context report`` used to print ``requested=40 allocated=10 used=23`` lines,
which made a 2% overflow and a 200% overflow look equally unremarkable. Each
source now gets a bar against its budget with an explicit marker when it
overflowed. Terminals without colour (``NO_COLOR``, no colour system, or
``PRIVATE_AGENT_ACCESSIBLE``) keep the text report.
"""

from __future__ import annotations

import os
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .console import resolve_style
from .theme import ACCESSIBLE_ENV

# Share bar width in cells; narrow enough to fit an 80-column terminal.
CONTEXT_BAR_WIDTH = 22


def _data(telemetry: Any) -> dict[str, Any]:
    if hasattr(telemetry, "to_dict"):
        return dict(telemetry.to_dict())
    return dict(telemetry or {})


def budget_bar(
    used: Any, budget: Any, *, width: int = CONTEXT_BAR_WIDTH
) -> ProgressBar:
    """Return a bar filled to ``used`` of ``budget``, coloured when overflowed."""
    try:
        used_value = int(used or 0)
        budget_value = int(budget or 0)
    except (TypeError, ValueError):
        used_value = budget_value = 0
    total = max(budget_value, 1)
    completed = min(max(used_value, 0), total)
    overflowed = used_value > budget_value
    fill = resolve_style("context.bar.over" if overflowed else "context.bar.fill")
    return ProgressBar(
        total=total,
        completed=completed,
        width=width,
        style=resolve_style("context.bar.back"),
        complete_style=fill,
        finished_style=fill,
    )


def _cell(value: Any, style: str | None = None) -> Text:
    return Text(str(value), style=resolve_style(style) if style else None)


def context_report_panel(
    telemetry: Any,
    *,
    bar_width: int = CONTEXT_BAR_WIDTH,
) -> Panel:
    """Render the last turn's per-source allocation as a bar table."""
    data = _data(telemetry)
    budgets: dict[str, Any] = data.get("budgets") or {}
    used: dict[str, Any] = data.get("used") or {}
    requested: dict[str, Any] = data.get("requested") or {}

    table = Table(
        box=None,
        show_header=True,
        header_style=resolve_style("muted"),
        padding=(0, 1),
    )
    table.add_column("Source", style=resolve_style("key"), no_wrap=True)
    table.add_column("Used", justify="right", style=resolve_style("value"))
    table.add_column("Budget", justify="right", style=resolve_style("value"))
    table.add_column("Share", no_wrap=True)

    for name in sorted(set(budgets) | set(used) | set(requested)):
        source_used = int(used.get(name, 0) or 0)
        source_budget = int(budgets.get(name, 0) or 0)
        used_cell = _cell(f"{source_used:,}")
        if source_used > source_budget:
            used_cell.append(" overflow", style=resolve_style("context.bar.over"))
        table.add_row(
            _cell(name),
            used_cell,
            _cell(f"{source_budget:,}" if source_budget else "unbudgeted"),
            budget_bar(source_used, source_budget, width=bar_width),
        )

    total_used = int(data.get("sum_used", 0) or 0)
    total_budget = int(data.get("sum_budgets", 0) or 0)
    window = int(data.get("total_tokens", 0) or 0)
    percent = float(data.get("usage_percent", 0.0) or 0.0)
    table.add_row(
        _cell("total", "muted"),
        _cell(f"{total_used:,}", "muted"),
        _cell(f"{window:,} window" if window else "no window", "muted"),
        _cell(f"{percent:.1f}% of window", "muted"),
    )

    subtitle = f"source budgets total {total_budget:,} tokens" if total_budget else None
    truncated = data.get("truncated") or []
    body: Any = table
    if truncated:
        body = Group(
            table,
            Text(
                "trimmed: " + ", ".join(str(name) for name in truncated),
                style=resolve_style("border.warning"),
            ),
        )

    return Panel(
        body,
        title=Text(
            f"Context usage ({data.get('profile', 'chat')})",
            style=resolve_style("accent"),
        ),
        subtitle=Text(subtitle) if subtitle else None,
        border_style=resolve_style("border.info"),
        padding=(0, 1),
    )


def bars_supported(console: Any) -> bool:
    """Return whether ``console`` should get bars instead of the text report."""
    if os.environ.get(ACCESSIBLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if console is None:
        return False
    if getattr(console, "no_color", False):
        return False
    return getattr(console, "color_system", None) is not None


__all__ = [
    "CONTEXT_BAR_WIDTH",
    "bars_supported",
    "budget_bar",
    "context_report_panel",
]
