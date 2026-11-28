"""Focused tests for the production full-screen chat surface."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from ui.chat_screen import ChatScreenUI
from ui.state import StatusState


def _render(layout) -> str:
    output = StringIO()
    Console(file=output, width=100, no_color=True).print(layout)
    return output.getvalue()


def test_chat_screen_has_conversation_input_and_live_ribbon_panes():
    console = Console(file=StringIO(), width=100, height=30, no_color=True)
    screen = ChatScreenUI(
        console,
        lambda: StatusState(
            model="local-model",
            context_tokens=4,
            context_maximum=10,
            permission_mode="auto",
        ),
    )
    screen.add_user("inspect the repository")
    screen.add_assistant("The repository is ready.")

    layout = screen.render()

    assert layout["chat"].name == "chat"
    assert layout["input"].name == "input"
    assert layout["ribbon"].name == "ribbon"
    rendered = _render(layout)
    for expected in (
        "Conversation",
        "inspect the repository",
        "The repository is ready.",
        "Input",
        "Live Status Ribbon",
        "Model: local-model",
        "Context: 40.0%",
        "Permission: auto",
    ):
        assert expected in rendered, expected


def test_chat_screen_keeps_user_text_literal():
    console = Console(file=StringIO(), width=100, height=30, no_color=True)
    screen = ChatScreenUI(console)
    screen.add_user("[bold]not markup[/bold]")

    rendered = _render(screen.render())

    assert "[bold]not markup[/bold]" in rendered


def test_chat_screen_show_renders_layout_without_clearing():
    output = StringIO()
    console = Console(file=output, width=100, height=30, no_color=True)
    screen = ChatScreenUI(console)
    screen.add_user("hello")

    screen.show()

    rendered = output.getvalue()
    assert "Conversation" in rendered
    assert "Input" in rendered
    assert "Live Status Ribbon" in rendered
    assert "hello" in rendered
