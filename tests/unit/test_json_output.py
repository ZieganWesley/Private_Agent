import io
import json

from ui import json_output


def test_emit_event_dict(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    json_output.emit_event({"a": 1})
    line = buf.getvalue().strip()
    assert json.loads(line) == {"a": 1}


def test_emit_event_dataclass(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class E:
        x: int = 5

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    json_output.emit_event(E())
    line = buf.getvalue().strip()
    assert json.loads(line) == {"x": 5}
