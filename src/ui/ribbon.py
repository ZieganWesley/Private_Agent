"""Canonical status ribbon formatter.

One function, one output: the single source of truth for the
``model=... | context=... | permission=... | ...`` line that used to be
printed (and re-printed, and re-styled) by several modules.

``status_ribbon`` returns the plain string (prompt-toolkit toolbars need one),
``ribbon_text`` returns the same words as a styled ``rich.text.Text`` so the
live footer can colour each segment by meaning instead of dimming the line.
"""

from __future__ import annotations

from rich.text import Text

from .console import resolve_style
from .state import StatusState
from .theme import SEGMENT_LABELS, SEGMENT_SEPARATOR

_PERMISSION_STYLES: dict[str, str] = {
    "readonly": "ribbon.permission.readonly",
    "read-only": "ribbon.permission.readonly",
    "read_only": "ribbon.permission.readonly",
    "locked": "ribbon.permission.readonly",
    "ask": "ribbon.permission.ask",
    "confirm": "ribbon.permission.ask",
    "confirm_each": "ribbon.permission.ask",
    "full": "ribbon.permission.full",
    "unrestricted": "ribbon.permission.full",
    "danger": "ribbon.permission.full",
}

_STATIC_STYLES: dict[str, str] = {
    "model": "ribbon.model",
    "mode": "ribbon.dim",
    "output": "ribbon.dim",
    "window": "ribbon.dim",
    "background": "ribbon.dim",
}


def context_segment_style(state: StatusState) -> str:
    """Return the style name for the context segment of ``state``."""
    percent = state.percent()
    if percent >= 80:
        return "ribbon.context.high"
    if percent >= 50:
        return "ribbon.context.warn"
    return "ribbon.context.ok"


def permission_segment_style(mode: object) -> str:
    """Return the style name for a permission mode."""
    key = str(mode or "auto").strip().lower()
    return _PERMISSION_STYLES.get(key, "ribbon.permission.auto")


def _segment_style(key: str, state: StatusState) -> str:
    if key == "context":
        return context_segment_style(state)
    if key == "permission":
        return permission_segment_style(state.permission_mode)
    if not key:
        return "ribbon.meta"
    return _STATIC_STYLES.get(key, "ribbon.dim")


def status_ribbon(
    state: StatusState, *, width: int | None = None, accessible: bool = False
) -> str:
    """Return the canonical ribbon for ``state`` as escaped markup-free text.

    ``width`` (optional) enables width-aware segment dropping for narrow
    terminals; essential segments (model/context/permission) are never dropped.
    """
    return SEGMENT_SEPARATOR.join(state.segments(width=width, accessible=accessible))


def ribbon_text(
    state: StatusState, *, width: int | None = None, accessible: bool = False
) -> Text:
    """Return the canonical ribbon as styled text, one span per segment."""
    text = Text()
    pairs = state.segment_pairs(width=width, accessible=accessible)
    for index, (key, value) in enumerate(pairs):
        if index:
            text.append(SEGMENT_SEPARATOR, style=resolve_style("ribbon.dim"))
        style = resolve_style(_segment_style(key, state))
        if key:
            label = SEGMENT_LABELS.get(key, key)
            text.append(f"{label}: ", style=resolve_style("ribbon.meta"))
        text.append(value, style=style)
    return text


__all__ = [
    "context_segment_style",
    "permission_segment_style",
    "ribbon_text",
    "status_ribbon",
]
