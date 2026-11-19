"""Shared SQLite lock for history tools."""

import threading

SQLITE_LOCK = threading.Lock()
