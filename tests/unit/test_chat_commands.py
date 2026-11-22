from types import SimpleNamespace

from agent import interactive


def test_format_conversation_export_renders_messages():
    history = [
        SimpleNamespace(type="human", content="hello"),
        SimpleNamespace(type="ai", content="hi there"),
    ]
    text = interactive.format_conversation_export(history)
    assert "# Conversation export" in text
    assert "## Human" in text
    assert "hello" in text
    assert "## Ai" in text
    assert "hi there" in text


def test_export_conversation_writes_file(tmp_path):
    history = [SimpleNamespace(type="human", content="hello")]
    destination = tmp_path / "out.md"
    saved = interactive.export_conversation(destination, history)
    assert saved == str(destination)
    assert destination.exists()
    assert "hello" in destination.read_text(encoding="utf-8")
