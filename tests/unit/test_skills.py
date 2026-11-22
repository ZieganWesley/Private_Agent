from unittest.mock import MagicMock

import pytest
from rich.console import Console

from tools import (
    create_skill,
    get_tool_context,
)

try:
    from agent import get_robust_chat_model
except ImportError:
    # Fallback definition if not explicitly exposed in agent
    def get_robust_chat_model(primary_model_name, fallback_model_name, tools=None):
        from langchain_ollama import ChatOllama

        try:
            model = ChatOllama(model=primary_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model
        except Exception:
            model = ChatOllama(model=fallback_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model


from skills import (
    AgentSkill,
    load_skills_from_folder,
    match_skill_by_relevancy,
)

console = Console()


def test_markdown_skill_loading(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()

    skill_file = skill_dir / "python_expert.md"
    skill_file.write_text(
        "# Python Expert\nWrite secure and robust python code.", encoding="utf-8"
    )

    empty_file = skill_dir / "empty_skill.md"
    empty_file.write_text("", encoding="utf-8")

    skills = load_skills_from_folder(str(skill_dir))
    assert "python_expert" in skills
    assert "empty_skill" in skills

    matched = match_skill_by_relevancy("I need help with python coding", skills)
    assert matched is not None
    assert matched.name == "Python Expert"


def test_create_skill_tool_writes_markdown_and_updates_active_registry(tmp_path):
    registry = {}
    context = get_tool_context()
    context.skill_directory = str(tmp_path / "skills")
    context.skill_registry = registry
    result = create_skill.invoke(
        {
            "name": "Python Testing",
            "description": "Help design focused Python tests.",
            "instructions": "Prefer pytest and cover important edge cases.",
        }
    )
    skill_file = tmp_path / "skills" / "python_testing.md"
    assert "Created skill 'Python Testing'" in result
    assert skill_file.is_file()
    content = skill_file.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "name: Python Testing" in content
    assert "description: Help design focused Python tests." in content
    assert "# Python Testing" in content
    assert "python_testing" in registry
    assert registry["python_testing"].name == "Python Testing"
    assert (
        match_skill_by_relevancy("I need help writing Python tests", registry)
        is registry["python_testing"]
    )


def test_create_skill_tool_rejects_path_traversal_and_existing_files(tmp_path):
    skills_dir = tmp_path / "skills"
    context = get_tool_context()
    context.skill_directory = str(skills_dir)
    context.skill_registry = {}
    args = {
        "name": "../outside",
        "description": "Unsafe path test.",
        "instructions": "Must not create outside the configured directory.",
    }
    assert "Error creating skill" in create_skill.invoke(args)
    assert not (tmp_path / "outside.md").exists()

    args["name"] = "safe_skill"
    assert "Created skill" in create_skill.invoke(args)
    original_content = (skills_dir / "safe_skill.md").read_text(encoding="utf-8")
    assert "already exists" in create_skill.invoke(args)
    assert (skills_dir / "safe_skill.md").read_text(
        encoding="utf-8"
    ) == original_content


@pytest.mark.asyncio
async def test_skill_creation_requires_interactive_approval(monkeypatch, tmp_path):
    import agent.runtime as agent
    import tools

    registry = {}
    context = get_tool_context()
    context.skill_directory = str(tmp_path / "skills")
    context.skill_registry = registry
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"create_skill": create_skill})
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent.console, "input", lambda prompt: "n")
    args = {
        "name": "requested_skill",
        "description": "A user-requested skill.",
        "instructions": "Apply the requested reusable guidance.",
    }
    result = await agent.execute_tool_call(
        {
            "name": "create_skill",
            "args": args,
            "id": "create-declined",
        }
    )
    assert "declined" in result.content
    assert not (tmp_path / "skills" / "requested_skill.md").exists()

    monkeypatch.setattr(agent.console, "input", lambda prompt: "y")
    result = await agent.execute_tool_call(
        {
            "name": "create_skill",
            "args": args,
            "id": "create-approved",
        }
    )
    assert "Created skill 'requested_skill'" in result.content
    assert "requested_skill" in registry
    skill_catalog = next(
        entry
        for entry in tools.describe_tool_catalog()
        if entry["name"] == "create_skill"
    )
    assert "interactive approval" in skill_catalog["permission"]


@pytest.mark.asyncio
async def test_skill_creation_is_blocked_without_interactive_terminal(
    monkeypatch, tmp_path
):
    import agent.runtime as agent

    context = get_tool_context()
    context.skill_directory = str(tmp_path / "skills")
    context.skill_registry = {}
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"create_skill": create_skill})
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: False))
    result = await agent.execute_tool_call(
        {
            "name": "create_skill",
            "args": {
                "name": "noninteractive",
                "description": "Not approved.",
                "instructions": "Must not be written.",
            },
            "id": "create-noninteractive",
        }
    )
    assert "requires interactive approval" in result.content
    assert not (tmp_path / "skills" / "noninteractive.md").exists()


def test_front_matter_overrides_name_and_description(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "release_notes.md").write_text(
        "---\n"
        "name: Release Notes\n"
        "description: Draft clear release notes for shipped changes.\n"
        "---\n"
        "# Release Notes\n\n"
        "Group changes by user impact.\n",
        encoding="utf-8",
    )

    skills = load_skills_from_folder(str(skill_dir))
    skill = skills["release_notes"]

    assert skill.name == "Release Notes"
    assert skill.description == "Draft clear release notes for shipped changes."
    assert "Group changes by user impact." in skill.system_prompt
    assert "description:" not in skill.system_prompt


def test_invalid_front_matter_falls_back_to_legacy_loading(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "broken_skill.md").write_text(
        "---\nname: [unterminated\n---\n# Broken Skill\nBody text.\n",
        encoding="utf-8",
    )

    skills = load_skills_from_folder(str(skill_dir))

    assert "broken_skill" in skills
    assert skills["broken_skill"].name == "Broken Skill"
    assert "Body text." in skills["broken_skill"].system_prompt


class _FakeEmbedder:
    def __init__(self):
        self.document_calls = 0

    @staticmethod
    def _vector(text):
        return [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class _BrokenEmbedder:
    def embed_documents(self, texts):
        raise RuntimeError("embedding backend unavailable")

    def embed_query(self, text):
        raise RuntimeError("embedding backend unavailable")


def test_embedding_matching_prefers_semantic_match_and_caches_skills():
    embedder = _FakeEmbedder()
    skills = {
        "python": AgentSkill("Python Expert", "Python coding", "Python workflow"),
        "writing": AgentSkill("Writing Coach", "Essays", "Writing workflow"),
    }

    first = match_skill_by_relevancy("Help with asyncio", skills, embedder)
    second = match_skill_by_relevancy("Python debugging", skills, embedder)

    assert first is skills["writing"]
    assert second is skills["python"]
    assert embedder.document_calls == 2


def test_embedding_failure_falls_back_to_keywords():
    skills = {
        "python": AgentSkill("Python Expert", "Python coding", "Python workflow"),
        "writing": AgentSkill("Writing Coach", "Essays", "Writing workflow"),
    }

    matched = match_skill_by_relevancy(
        "I need Python coding help",
        skills,
        _BrokenEmbedder(),
    )

    assert matched is skills["python"]
