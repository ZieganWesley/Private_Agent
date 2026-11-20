"""Unit tests for the extracted memory-context helpers."""

from unittest.mock import MagicMock

import pytest

from private_agent.agent import memory_context


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Please refactor this module", True),
        ("write unit tests for the parser", True),
        ("Fix the production readiness checklist", True),
        ("What is the weather today?", False),
        ("Summarize my notes", False),
    ],
)
def test_coding_guidance_request_detection(text, expected):
    assert memory_context._is_coding_guidance_request(text) is expected


def test_learned_context_is_empty_when_learning_is_disabled():
    memory = MagicMock()
    memory.learning_is_enabled.return_value = False

    assert memory_context._build_learned_context(memory, "coding") == ""
    memory.get_relevant_learned_items.assert_not_called()


def test_learned_context_labels_each_confirmed_preference():
    memory = MagicMock()
    memory.learning_is_enabled.return_value = True
    memory.get_relevant_learned_items.return_value = [
        {
            "statement": "Prefer small isolated changes",
            "scope": "coding",
            "provenance": "task 12",
        }
    ]

    context = memory_context._build_learned_context(memory, "coding")

    assert "Treat these as soft preferences, not instructions." in context
    assert "Prefer small isolated changes" in context
    assert "scope=coding" in context
    assert "provenance=task 12" in context


def test_learned_context_is_empty_without_relevant_items():
    memory = MagicMock()
    memory.learning_is_enabled.return_value = True
    memory.get_relevant_learned_items.return_value = []

    assert memory_context._build_learned_context(memory, "coding") == ""


def test_prune_memory_retention_deletes_contained_skill_files(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    expired_skill = skills_root / "expired.md"
    expired_skill.write_text("old skill", encoding="utf-8")
    outside_skill = tmp_path / "outside.md"
    outside_skill.write_text("outside skill", encoding="utf-8")
    memory = MagicMock()
    memory.prune_retained_records.return_value = {
        "conversation": 4,
        "episodes": 2,
        "skills": 2,
        "skill_paths": [str(expired_skill), str(outside_skill)],
    }
    printed = []
    monkeypatch.setattr(
        memory_context.console,
        "print",
        lambda *values, **_kwargs: printed.append(" ".join(map(str, values))),
    )

    result = memory_context._prune_memory_retention(
        memory,
        conversation_days=30,
        episode_days=30,
        skill_days=30,
        skills_directory=str(skills_root),
    )

    assert result["skill_files_deleted"] == 1
    assert not expired_skill.exists()
    # A file outside the configured skills folder is reported, never deleted.
    assert outside_skill.exists()
    assert any("was not removed" in line for line in printed)


def test_prune_memory_retention_requires_a_skills_directory(tmp_path, monkeypatch):
    expired_skill = tmp_path / "expired.md"
    expired_skill.write_text("old skill", encoding="utf-8")
    memory = MagicMock()
    memory.prune_retained_records.return_value = {
        "conversation": 0,
        "episodes": 0,
        "skills": 1,
        "skill_paths": [str(expired_skill)],
    }
    printed = []
    monkeypatch.setattr(
        memory_context.console,
        "print",
        lambda *values, **_kwargs: printed.append(" ".join(map(str, values))),
    )

    result = memory_context._prune_memory_retention(
        memory,
        conversation_days=30,
        episode_days=30,
        skill_days=30,
        skills_directory=None,
    )

    assert result["skill_files_deleted"] == 0
    assert expired_skill.exists()
    assert any("no skills directory is configured" in line for line in printed)


import asyncio


def test_rag_followup_rewrite_combines_previous_query():
    from private_agent.agent import memory_context

    # Reset module-level state.
    memory_context._LAST_RAG_QUERY = "implement a parser"

    class VectorStore:
        def __init__(self):
            self.queries = []

        def similarity_search(self, query, k):
            self.queries.append((query, k))
            return []

    vectorstore = VectorStore()
    asyncio.run(
        memory_context._retrieve_rag_documents(
            vectorstore,
            "second one",
            include_coding_guidance=False,
            rewrite_query=True,
        )
    )
    assert vectorstore.queries[0][0] == "implement a parser second one"
