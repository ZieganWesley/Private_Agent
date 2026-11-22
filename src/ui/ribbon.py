"""Canonical status ribbon formatter.

One function, one output: the single source of truth for the
``model=... | context=... | permission=... | ...`` line that used to be
printed (and re-printed, and re-styled) by several modules.
"""

from __future__ import annotations

from .state import StatusState
from .theme import SEGMENT_SEPARATOR


def status_ribbon(
    state: StatusState, *, width: int | None = None, accessible: bool = False
) -> str:
    """Return the canonical ribbon for ``state``.

    ``width`` (optional) enables width-aware segment dropping for narrow
    terminals; essential segments (model/context/permission) are never dropped.
    """
    return SEGMENT_SEPARATOR.join(state.segments(width=width, accessible=accessible))


__all__ = ["status_ribbon"]
