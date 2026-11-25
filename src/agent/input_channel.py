"""One stdin owner for the session and a background steering reader.

The interactive prompt owns stdin between turns.  During a turn this channel
reads lines in a daemon thread and pushes them to the controller.  Permission
prompts call :meth:`pause` and :meth:`resume` so they never compete for input.
Non-TTY sessions are inert unless a line reader is supplied explicitly.
"""

from __future__ import annotations

import os
import select
import sys
import threading
from collections.abc import Callable

from .turn_control import TurnController


class InputChannel:
    """Read steering lines while a turn is active."""

    def __init__(
        self,
        controller: TurnController,
        *,
        read_line: Callable[[], str] | None = None,
        is_tty: bool | None = None,
        enabled: bool = True,
    ):
        self.controller = controller
        self._read_line = read_line
        self._is_tty = sys.stdin.isatty() if is_tty is None else bool(is_tty)
        self._fd = self._stdin_fd() if read_line is None else None
        self._enabled = enabled and self._is_tty and (
            read_line is not None or self._fd is not None
        )
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _stdin_fd() -> int | None:
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            return None
        return fd if isinstance(fd, int) and fd >= 0 else None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if not self._enabled or self.running:
            return
        self._stop.clear()
        self._paused.clear()
        target = self._run_fd if self._fd is not None else self._run
        self._thread = threading.Thread(
            target=target,
            name="private-agent-steering",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        assert self._read_line is not None
        while not self._stop.is_set():
            try:
                line = self._read_line()
            except (EOFError, OSError, ValueError):
                break
            if line == "":
                break
            if self._paused.is_set():
                continue
            self.submit(line)

    def _run_fd(self) -> None:
        """Read stdin only when ready, so stop never leaves a blocked reader."""
        assert self._fd is not None
        buffer = b""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
            except (OSError, ValueError):
                break
            if self._stop.is_set():
                break
            if not ready:
                continue
            try:
                chunk = os.read(self._fd, 4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").rstrip("\r")
                if not self._paused.is_set():
                    self.submit(line)
        self._thread = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def submit(self, text: str) -> None:
        """Parse one line and route it to the controller."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        lowered = cleaned.lower()
        if lowered == "/stop":
            self.controller.request_stop()
            return
        if lowered.startswith("/steer"):
            instruction = cleaned[len("/steer") :].strip()
            if instruction:
                self.controller.queue_message(instruction, mode="interrupt")
            return
        self.controller.queue_message(cleaned, mode="queue", source="input")

    def stop(self) -> None:
        self._stop.set()
        self._paused.clear()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.5)
        self._thread = None


__all__ = ["InputChannel"]
