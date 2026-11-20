"""Focused tests for ToolContext as the single tool-state source (B3)."""

from dataclasses import replace

import pytest

from private_agent import code_tasks
from private_agent.tools import (
    ToolContext,
    clear_schema_reads,
    get_tool_context,
    set_tool_context,
)


@pytest.fixture(autouse=True)
def restore_tool_context():
    """Restore the shared context after each test."""
    context = get_tool_context()
    original = replace(
        context,
        schema_read_tables=set(context.schema_read_tables or set()),
    )
    yield context
    set_tool_context(original)


def test_get_tool_context_returns_shared_instance():
    assert get_tool_context() is get_tool_context()


def test_set_tool_context_copies_all_fields():
    context = ToolContext(
        db_path="/tmp/example.sqlite",
        task_session_id="session",
        task_id="task",
        skill_directory="/tmp/skills",
        skill_registry={"skill": object()},
        skill_session_id="skill-session",
        code_task=object(),
        runtime_owner_id="owner",
        schema_read_tables={"chat_history"},
        enabled_packs={"Coding"},
    )

    set_tool_context(context)
    current = get_tool_context()

    assert current.db_path == "/tmp/example.sqlite"
    assert current.task_session_id == "session"
    assert current.task_id == "task"
    assert current.skill_directory == "/tmp/skills"
    assert current.skill_registry == context.skill_registry
    assert current.skill_session_id == "skill-session"
    assert current.code_task is context.code_task
    assert current.runtime_owner_id == "owner"
    assert current.schema_read_tables == {"chat_history"}
    assert current.enabled_packs == {"Coding"}


def test_set_tool_context_keeps_runtime_owner_and_schema_reads_when_unset():
    current = get_tool_context()
    current.runtime_owner_id = "owner"
    current.schema_read_tables = {"agent_tasks"}

    set_tool_context(
        ToolContext(
            db_path=current.db_path,
            runtime_owner_id=None,
            schema_read_tables=None,
        )
    )

    assert get_tool_context().runtime_owner_id == "owner"
    assert get_tool_context().schema_read_tables == {"agent_tasks"}


def test_clear_schema_reads_empties_shared_set():
    get_tool_context().schema_read_tables.update({"agent_tasks", "chat_history"})

    clear_schema_reads()

    assert get_tool_context().schema_read_tables == set()


def test_code_task_tools_read_code_task_from_context(monkeypatch):
    class FakeCodeTask:
        def run_tests(self):
            return "tests passed"

        def checkpoint(self, label, paths):
            return f"checkpoint {label}: {paths}"

        def finalize(self):
            return "finalized"

    monkeypatch.setattr(get_tool_context(), "code_task", FakeCodeTask())

    assert code_tasks.run_project_unit_tests.invoke({}) == "tests passed"
    assert (
        code_tasks.checkpoint_code_task.invoke(
            {"label": "step", "paths": ["src/app.py"]}
        )
        == "checkpoint step: ['src/app.py']"
    )
    assert code_tasks.finalize_code_task.invoke({}) == "finalized"


def test_code_task_tools_degrade_when_no_task_is_active(monkeypatch):
    monkeypatch.setattr(get_tool_context(), "code_task", None)

    assert (
        "No isolated code task is active"
        in code_tasks.run_project_unit_tests.invoke({})
    )
    assert "No isolated code task is active" in code_tasks.checkpoint_code_task.invoke(
        {"label": "step", "paths": ["src/app.py"]}
    )
    assert "No isolated code task is active" in code_tasks.finalize_code_task.invoke({})
