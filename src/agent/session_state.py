"""Mutable session state shared between the runtime loop and its helpers."""

from __future__ import annotations

from typing import Any

from .session_stats import SessionStats

SESSION_STATS = SessionStats()
# Descriptive fields for the exit statistics table; cleared per session.
SESSION_INFO: dict[str, Any] = {}
