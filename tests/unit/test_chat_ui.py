from io import StringIO

from rich.console import Console

from private_agent.chat_ui import ChatTranscript, render_chat_message


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
