"""Startup framing renderables: section rules and key/value summaries."""

from __future__ import annotations

import io

from rich.console import Console

from ui.startup import (
    capability_icons,
    field_text,
    figlet_banner,
    key_value_columns,
    ready_header,
    runtime_info_panel,
    startup_rule,
    startup_section,
)


def _render(renderable, width: int = 80) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=width, no_color=True).print(renderable)
    return buffer.getvalue()


def test_banner_renders_ascii_art_and_subtitle():
    rendered = _render(figlet_banner(), width=100)

    assert "Local Inference & Tool Runtime" in rendered
    assert "\u2502" in rendered or "\u2503" in rendered
    # Pyfiglet's slant font draws the title; the plain title is not printed.
    assert "Private Agent" not in rendered


def test_startup_rule_names_the_section():
    rendered = _render(startup_rule("Configuration"))
    assert "Configuration" in rendered
    assert "\u2500" in rendered


def test_every_field_reaches_the_summary():
    rendered = _render(
        startup_section(
            "Configuration",
            (
                ("Config file", "/tmp/private_config.json"),
                ("RAG documents", "not configured"),
                ("Skills", "/tmp/skills"),
            ),
        )
    )
    for expected in (
        "Config file:",
        "/tmp/private_config.json",
        "RAG documents:",
        "not configured",
        "Skills:",
        "/tmp/skills",
    ):
        assert expected in rendered, expected


def test_long_paths_fold_instead_of_truncating():
    path = "/home/someone/projects/private-agent/private_config.json"
    rendered = _render(
        startup_section("Configuration", (("Config file", path),)),
        width=40,
    )
    assert "\u2026" not in rendered
    assert "".join(rendered.split()).count("private_config.json") == 1
    assert max(len(line) for line in rendered.splitlines()) <= 40


def test_field_text_never_treats_a_value_as_markup():
    rendered = _render(key_value_columns((("Model", "[bold]evil[/bold]"),)))
    assert "[bold]evil[/bold]" in rendered


def test_summary_renders_on_a_console_without_the_theme():
    # A bare Console does not know the application theme: unknown style names
    # would raise MissingStyle, so every lookup must carry a fallback.
    rendered = _render(field_text("Model", "m"), width=20)
    assert "Model: m" in rendered


def test_runtime_info_panel_shows_runtime_facts_and_capability_icons():
    rendered = _render(
        runtime_info_panel(
            model="local-model",
            provider="Local (llama.cpp)",
            permission="Auto",
            isolation="Available — bubblewrap + prlimit",
            workspace="/work/project",
            rag="/work/rag",
            rag_active=True,
            skills=3,
            capabilities={
                "tools": "yes",
                "function_calls": "unknown",
                "structured_output": "no",
                "thinking": "yes",
                "vision": "no",
                "audio": "unknown",
            },
            platform="Linux (x86_64)",
        ),
        width=120,
    )

    for expected in (
        "Model & Runtime Info",
        "local-model",
        "Local (llama.cpp)",
        "Auto",
        "Available — bubblewrap + prlimit",
        "Linux (x86_64)",
        "/work/project",
        "/work/rag",
        "active",
        "3 · 🔧❓➖🧠🙈🔇",
        "Commands: /help · /mode · /think · /net · /up · /down · switch · exit",
    ):
        assert expected in rendered, expected


def test_runtime_info_panel_keeps_values_literal():
    rendered = _render(
        runtime_info_panel(
            model="[bold]evil[/bold]",
            provider="Local",
            permission="Full",
            isolation="unavailable",
            workspace="[red]/work[/red]",
            rag="[not-a-style]/rag",
            rag_active=False,
            skills=0,
            capabilities={},
        ),
        width=120,
    )

    assert "[bold]evil[/bold]" in rendered
    assert "[red]/work[/red]" in rendered
    assert "[not-a-style]/rag" in rendered
    assert "🔧❓➖🧠🙈🔇" not in rendered
    assert "🚫➖➖➖🙈🔇" in rendered


def test_capability_icons_uses_unknown_and_no_states():
    assert capability_icons(None) == "🚫➖➖➖🙈🔇"
    assert capability_icons({"tools": "unknown", "vision": "yes"}) == "🛠️➖➖➖👁️🔇"


def test_ready_header_combines_banner_panel_and_command_footer():
    rendered = _render(
        ready_header(
            model="local-model",
            provider="Local (llama.cpp)",
            permission="Auto",
            isolation="Available — bubblewrap + prlimit",
            workspace="/work/project",
            rag="/work/rag",
            rag_active=True,
            skills=3,
            capabilities={"tools": "yes"},
            platform="Linux (x86_64)",
        ),
        width=120,
    )

    for expected in (
        "Local Inference & Tool Runtime",
        "Model & Runtime Info",
        "local-model",
        "Commands: /help · /mode · /think · /net · /up · /down · switch · exit",
    ):
        assert expected in rendered, expected
