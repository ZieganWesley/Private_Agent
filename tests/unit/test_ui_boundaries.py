"""Boundary guards for the UI package (Part B7)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ui import SessionUI


def test_gauge_glyphs_and_tokens_only_live_under_ui():
    root = Path("src")
    ui_dir = root / "ui"
    forbidden_tokens = ["\u25b0", "\u25b1", "GAUGE_FILLED", "GAUGE_EMPTY"]
    offenders = []
    for path in root.rglob("*.py"):
        if ui_dir in path.parents or path == ui_dir:
            continue
        # skip bytecode caches
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                offenders.append((str(path), token))
    assert not offenders, f"gauge tokens outside ui/: {offenders}"

    # context_bar must not be called outside ui/.
    call_offenders = []
    for path in root.rglob("*.py"):
        if ui_dir in path.parents or path == ui_dir:
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "context_bar(" in text:
            call_offenders.append(str(path))
    assert not call_offenders, f"context_bar() calls outside ui/: {call_offenders}"


def test_runtime_has_no_old_chrome_labels():
    text = Path("src/agent/runtime.py").read_text(encoding="utf-8")
    for label in ("[Latency]", "[Generation]", "Working", "[Context State]"):
        assert label not in text, f"old chrome label {label!r} still in runtime.py"
    assert 'console.print(f"[dim]model=' not in text


def test_scripted_turn_prints_one_turn_line_with_ribbon_last(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    console = Console(record=True, width=200)
    ui = SessionUI(console=console)
    ui.bind(model="m", permission="auto", output_tokens=10)
    ui.set_context(2, 10)
    report = ui.turn_started("hi")
    ui.turn_finished(report)
    text = console.export_text()
    assert text.count("[Turn]") == 1
    assert "ETA" not in text
    assert "Working" not in text
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, "no output produced"
    assert "[Turn]" in lines[-1], f"turn summary not last: {lines[-1]!r}"
    assert any("Model: m" in line for line in lines), "startup ribbon missing"
