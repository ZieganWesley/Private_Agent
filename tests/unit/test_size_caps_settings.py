"""Tool/search/citation/compaction caps come from settings (K4/14.6)."""

from __future__ import annotations

import pytest

import config
from config import DEFAULT_CONFIG
from config.settings import Settings

GIB = 1024**3
MIB = 1024**2


def _settings(agent: dict | None = None, tools: dict | None = None) -> Settings:
    return Settings.from_config(
        {"agent": agent or {}, "tools": tools or {}},
        default_paths=DEFAULT_CONFIG["paths"],
    )


def test_caps_default_to_the_previous_fixed_values() -> None:
    settings = _settings()

    assert settings.tool_result_preview_chars == 500
    assert settings.tool_argument_preview_chars == 200
    assert settings.search_max_files == 10000
    assert settings.search_max_bytes == 16 * MIB
    assert settings.search_output_chars == 12000
    assert settings.task_citation_budget_chars == 10000


def test_compaction_source_cap_is_derived_from_the_context_window() -> None:
    assert (
        _settings({"max_context_tokens": 12000}).context_compaction_source_tokens
        == 6000
    )
    assert (
        _settings({"max_context_tokens": 30000}).context_compaction_source_tokens
        == 15000
    )
    assert (
        _settings({"max_context_tokens": 1000}).context_compaction_source_tokens == 1000
    )
    assert (
        _settings(
            {"max_context_tokens": 30000, "compaction_source_tokens": 9000}
        ).context_compaction_source_tokens
        == 9000
    )


def test_caps_can_be_configured() -> None:
    settings = _settings(
        {
            "tool_result_preview_chars": 1200,
            "tool_argument_preview_chars": 80,
            "compaction_source_tokens": 4000,
        },
        {
            "search_max_files": 500,
            "search_max_bytes": 2 * MIB,
            "search_output_chars": 24000,
            "task_citation_budget_chars": 20000,
        },
    )

    assert settings.tool_result_preview_chars == 1200
    assert settings.tool_argument_preview_chars == 80
    assert settings.context_compaction_source_tokens == 4000
    assert settings.search_max_files == 500
    assert settings.search_max_bytes == 2 * MIB
    assert settings.search_output_chars == 24000
    assert settings.task_citation_budget_chars == 20000


def test_caps_below_their_floor_fall_back_to_defaults() -> None:
    settings = _settings(
        {"tool_result_preview_chars": 10, "tool_argument_preview_chars": 1},
        {"search_max_files": 1, "search_max_bytes": 1, "search_output_chars": 1},
    )

    assert settings.tool_result_preview_chars == 500
    assert settings.tool_argument_preview_chars == 200
    assert settings.search_max_files == 10000
    assert settings.search_max_bytes == 16 * MIB
    assert settings.search_output_chars == 12000


def test_module_constants_mirror_settings() -> None:
    assert config.SETTINGS.tool_result_preview_chars == config.TOOL_RESULT_PREVIEW_CHARS
    assert (
        config.SETTINGS.tool_argument_preview_chars
        == config.TOOL_ARGUMENT_PREVIEW_CHARS
    )
    assert config.SETTINGS.search_max_files == config.SEARCH_MAX_FILES
    assert config.SETTINGS.search_max_bytes == config.SEARCH_MAX_BYTES
    assert config.SETTINGS.search_output_chars == config.SEARCH_OUTPUT_CHARS
    assert (
        config.SETTINGS.task_citation_budget_chars == config.TASK_CITATION_BUDGET_CHARS
    )
    assert (
        config.SETTINGS.context_compaction_source_tokens
        == config.CONTEXT_COMPACTION_SOURCE_TOKENS
    )


def test_transcript_uses_the_configured_source_cap(monkeypatch) -> None:
    from langchain_core.messages import HumanMessage

    from agent import compaction
    from agent.compaction import _transcript

    messages = [HumanMessage(content="word " * 400) for _ in range(4)]

    monkeypatch.setattr(compaction, "CONTEXT_COMPACTION_SOURCE_TOKENS", 25)
    capped = _transcript(messages)
    assert compaction.token_count(capped) <= 25

    assert compaction.token_count(_transcript(messages, 10)) <= 10


def test_workspace_search_output_respects_the_configured_cap(
    temp_workspace, monkeypatch
) -> None:
    from tools import filesystem, search_workspace_text

    content = "\n".join(f"needle {index}" for index in range(200))
    (temp_workspace / "many.txt").write_text(content, encoding="utf-8")

    monkeypatch.setattr(filesystem, "_MAX_SEARCH_OUTPUT_CHARS", 80)
    result = search_workspace_text.invoke({"query": "needle", "limit": 100})

    assert "[Search output truncated.]" in result
    assert len(result) <= 80 + len("\n[Search output truncated.]")


def test_citation_budget_is_enforced_from_the_configured_value(monkeypatch) -> None:
    from database import tasks

    references = ["x" * 1000, "y" * 1000]

    monkeypatch.setattr(tasks, "PLAN_MAX_CITATION_CHARS", 1500)
    with pytest.raises(ValueError, match="1500-character plan budget"):
        tasks.TaskMixin._normalize_research_references(references)

    monkeypatch.setattr(tasks, "PLAN_MAX_CITATION_CHARS", 2000)
    assert tasks.TaskMixin._normalize_research_references(references) == references
