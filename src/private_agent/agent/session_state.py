"""Mutable session state shared between the runtime loop and its helpers."""

from __future__ import annotations

from .session_stats import SessionStats

SESSION_STATS = SessionStats()
