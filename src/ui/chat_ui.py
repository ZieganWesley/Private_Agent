"""Lightweight chat-style transcript rendering helpers.

These helpers avoid full-screen redraws: each message or streamed chunk is
written incrementally to the shared console.
"""

from __future__ import annotations

from typing import Any

from .console import console as _shared_console


class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: Any = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp

    @property
    def avatar(self) -> str:
        return "You" if self.role.lower() in {"user", "you"} else "Agent"

    def render_text(self) -> str:
        return f"{self.avatar}: {self.content}"


class ChatTranscript:
    def __init__(self, console=None):
        self.console = console or _shared_console
        self.messages: list[ChatMessage] = []

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Any = None,
        markdown: bool = False,
        echo: bool = True,
    ) -> ChatMessage:
        message = ChatMessage(role, content, timestamp)
        self.messages.append(message)
        if not echo:
            return message
        if markdown:
            self.console.print(f"[bold green]{message.avatar}:[/bold green]")
            from rich.markdown import Markdown

            self.console.print(Markdown(content), soft_wrap=True)
        else:
            self.console.print(message.render_text())
        return message

    def stream_chunk(self, role: str, chunk: str, end: str = "") -> None:
        # Incremental, non-blocking write for streaming tokens.
        self.console.print(chunk, end=end)


def render_chat_message(
    role: str,
    content: str,
    timestamp: Any = None,
    console=None,
    markdown: bool = False,
    echo: bool = True,
) -> str:
    message = ChatMessage(role, content, timestamp)
    target = console or _shared_console
    if not echo:
        return message.render_text()
    if markdown:
        target.print(f"[bold green]{message.avatar}:[/bold green]")
        from rich.markdown import Markdown

        target.print(Markdown(content), soft_wrap=True)
    else:
        target.print(message.render_text())
    return message.render_text()


# Backwards/alternate friendly aliases.
ChatUI = ChatTranscript
render_message = render_chat_message
stream_chunk = ChatTranscript.stream_chunk
