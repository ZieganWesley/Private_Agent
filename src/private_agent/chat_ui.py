"""Lightweight chat-style transcript rendering helpers.

These helpers avoid full-screen redraws: each message or streamed chunk is
written incrementally to the shared console.
"""

from __future__ import annotations

import datetime
from typing import Any

from .console import console as _shared_console


class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: Any = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.datetime.now()

    @property
    def avatar(self) -> str:
        return "You" if self.role.lower() in {"user", "you"} else "Agent"

    def render_text(self) -> str:
        stamp = self.timestamp.strftime("%H:%M:%S") if hasattr(self.timestamp, "strftime") else str(self.timestamp)
        return f"[{stamp}] {self.avatar}: {self.content}"


class ChatTranscript:
    def __init__(self, console=None):
        self.console = console or _shared_console
        self.messages: list[ChatMessage] = []

    def add_message(self, role: str, content: str, timestamp: Any = None) -> ChatMessage:
        message = ChatMessage(role, content, timestamp)
        self.messages.append(message)
        self.console.print(message.render_text())
        return message

    def stream_chunk(self, role: str, chunk: str, end: str = "") -> None:
        # Incremental, non-blocking write for streaming tokens.
        self.console.print(chunk, end=end)


def render_chat_message(role: str, content: str, timestamp: Any = None, console=None) -> str:
    message = ChatMessage(role, content, timestamp)
    (console or _shared_console).print(message.render_text())
    return message.render_text()


# Backwards/alternate friendly aliases.
ChatUI = ChatTranscript
render_message = render_chat_message
stream_chunk = ChatTranscript.stream_chunk
