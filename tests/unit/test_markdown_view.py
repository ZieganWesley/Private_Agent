"""One Markdown renderer for streamed and complete replies."""

from __future__ import annotations

import io

from rich.console import Console

from ui.chat_ui import ChatTranscript
from ui.markdown_view import MarkdownStream, markdown_renderable, print_markdown

FENCED = "```python\ndef answer():\n    return 42\n```\n"


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=80, no_color=True), buffer


def test_complete_reply_renders_the_fence_once():
    console, buffer = _console()
    print_markdown(console, FENCED)
    rendered = buffer.getvalue()
    assert "def answer()" in rendered
    assert "```" not in rendered


def test_stream_repaints_once_for_a_burst_then_once_on_stop():
    console, buffer = _console()
    stream = MarkdownStream(console, refresh_per_second=8)
    for chunk in ("Hel", "lo ", "wor", "ld"):
        stream.update(chunk)
    assert stream.renders == 1, "a burst of chunks was parsed once per chunk"
    stream.stop()
    assert stream.renders == 2
    assert "Hello world" in buffer.getvalue()
    assert stream.text == "Hello world"


def test_stream_keeps_a_fence_whole_when_it_arrives_in_pieces():
    console, buffer = _console()
    stream = MarkdownStream(console)
    for piece in ("```pyth", "on\ndef ans", "wer():\n    return 42\n", "```\n"):
        stream.update(piece)
    stream.stop()
    rendered = buffer.getvalue()
    assert "def answer()" in rendered
    assert "```" not in rendered


def test_empty_stream_paints_nothing():
    console, buffer = _console()
    stream = MarkdownStream(console)
    stream.update("")
    stream.stop()
    assert stream.renders == 0
    assert buffer.getvalue() == ""


def test_transcript_stream_chunk_uses_the_shared_renderer():
    console, buffer = _console()
    transcript = ChatTranscript(console=console)
    transcript.stream_chunk("assistant", FENCED)
    rendered = buffer.getvalue()
    assert "def answer()" in rendered
    assert "```" not in rendered


def test_renderable_is_a_markdown_document():
    assert markdown_renderable("# Title").markup.startswith("# Title")
