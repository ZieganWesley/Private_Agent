from io import StringIO

from rich.console import Console

from private_agent.ui.chat_ui import ChatTranscript, render_chat_message


def test_chat_transcript_renders_messages():
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    transcript = ChatTranscript(console=console)
    transcript.add_message("user", "hello")
    transcript.add_message("assistant", "hi")
    output = buffer.getvalue()
    assert "You: hello" in output
    assert "Agent: hi" in output


def test_render_chat_message_returns_text():
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    text = render_chat_message("user", "ping", console=console)
    assert "You: ping" in text


def test_chat_transcript_render_budget():
    import time

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    transcript = ChatTranscript(console=console)
    start = time.monotonic()
    for index in range(500):
        transcript.add_message("assistant", f"message {index}")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_chat_ui_wired_into_interactive_flow():
    from pathlib import Path

    source = Path("src/private_agent/agent/runtime.py").read_text(encoding="utf-8")
    assert "ChatTranscript" in source
    assert "chat_transcript.add_message" in source
    assert "chat_transcript.stream_chunk" in source


def test_chat_transcript_stream_chunk_writes_incrementally():
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    transcript = ChatTranscript(console=console)
    transcript.stream_chunk("assistant", "hello", end="")
    assert "hello" in buffer.getvalue()
