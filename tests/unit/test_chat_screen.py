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
            provider="Local (llama.cpp)",
            thinking_enabled=True,
            skills_count=3,
            capabilities={"tools": "yes"},
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
        "Auto-scroll active · History: 2 block(s)",
        "User: inspect the repository",
        "AI:",
        "The repository is ready.",
        "You ›",
        "Live Status Ribbon",
        "[Model & Mode] Local (llama.cpp)/local-model",
        "Mode:agentic",
        "🧠Think:ON(med)",
        "[Safety & Execution] 🔐Perm:auto",
        "🧰Tools:0",
        "⚙️BG:0",
        "🌐Net:allow",
        "[Context & Caps] 📊Ctx:4/10 (40%)",
        "📁WS:on",
        "🔎RAG:on",
        "📚Skills:3",
        "Caps:🔧➖➖➖🙈🔇",
        "[● ready]",
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
    assert "You ›" in rendered
    assert "Live Status Ribbon" in rendered
    assert "User: hello" in rendered


def test_chat_screen_busy_state_matches_reference_ribbon():
    console = Console(file=StringIO(), width=100, height=30, no_color=True)
    screen = ChatScreenUI(console)
    screen.add_user("hello")
    screen.set_processing(True)

    rendered = _render(screen.render())

    assert "Model is responding..." in rendered
    assert "⏳ busy" in rendered
    assert "You ›" not in rendered
