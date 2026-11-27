"""Rich menus: panel layout and numbered/value selection."""

from __future__ import annotations

import io

from rich.console import Console

from ui.menu import MenuOption, menu_panel, select_menu


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=80, no_color=True), buffer


def test_menu_panel_numbers_labels_and_hints():
    console, buffer = _console()
    console.print(
        menu_panel(
            "Local Inference Control",
            [
                MenuOption("1", "Start inference server", "llama serve"),
                ("0", "Exit", "Quit CLI"),
            ],
        )
    )

    rendered = buffer.getvalue()
    assert "Local Inference Control" in rendered
    assert "[1] Start inference server  llama serve" in rendered
    assert "[2] Exit  Quit CLI" in rendered


def test_select_menu_accepts_the_displayed_number():
    console, buffer = _console()

    picked = select_menu(
        "Pick",
        [("alpha", "Alpha"), ("beta", "Beta")],
        console=console,
        input_fn=lambda _prompt: "2",
    )

    assert picked == "beta"
    assert "Alpha" in buffer.getvalue()


def test_select_menu_accepts_an_exact_text_value():
    console, _buffer = _console()

    picked = select_menu(
        "Permission",
        [("manual", "Manual"), ("auto", "Auto")],
        console=console,
        input_fn=lambda _prompt: "manual",
    )

    assert picked == "manual"


def test_select_menu_uses_the_default_for_an_empty_answer():
    console, _buffer = _console()

    picked = select_menu(
        "Models",
        [("small", "Small"), ("large", "Large")],
        console=console,
        input_fn=lambda _prompt: "",
        default="large",
    )

    assert picked == "large"


def test_select_menu_returns_a_custom_model_id_when_allowed():
    console, _buffer = _console()

    picked = select_menu(
        "Models",
        [("gpt-4o", "gpt-4o")],
        console=console,
        input_fn=lambda _prompt: "my-custom-model",
        allow_custom=True,
    )

    assert picked == "my-custom-model"


def test_select_menu_invalid_answer_falls_back_to_the_default():
    console, buffer = _console()

    picked = select_menu(
        "Models",
        [("one", "One")],
        console=console,
        input_fn=lambda _prompt: "not-a-choice",
        default="one",
    )

    assert picked == "one"
    assert "Invalid choice" in buffer.getvalue()
