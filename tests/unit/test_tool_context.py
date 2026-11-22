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

_DURABLE_FIELDS = (
    "task_session_id",
    "task_id",
    "code_task",
    "skill_directory",
    "skill_registry",
    "skill_session_id",
    "subagent_runner",
    "interactive",
)


@pytest.fixture(autouse=True)
def restore_tool_context():
    """Restore the shared context after each test."""
    context = get_tool_context()
    durable = {name: getattr(context, name) for name in _DURABLE_FIELDS}
    task_ids = dict(context.task_ids)
    enabled_packs = set(context.enabled_packs)
    original = replace(
        context,
        schema_read_tables=set(context.schema_read_tables or set()),
    )
    yield context
    # Turns carry durable state back into the shared context, so restore it too.
    for name, value in durable.items():
        setattr(context, name, value)
    context.task_ids.clear()
    context.task_ids.update(task_ids)
    context.enabled_packs.clear()
    context.enabled_packs.update(enabled_packs)
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


@pytest.mark.asyncio
async def test_turn_tool_context_isolates_enabled_packs():
    import asyncio

    from private_agent.tools import begin_turn_tool_context, reset_turn_tool_context

    async def turn(pack):
        token = begin_turn_tool_context()
        try:
            get_tool_context().enabled_packs.add(pack)
            await asyncio.sleep(0)
            return set(get_tool_context().enabled_packs)
        finally:
            reset_turn_tool_context(token)

    assert await asyncio.gather(turn("Coding"), turn("Research")) == [
        {"Coding"},
        {"Research"},
    ]
    assert get_tool_context().enabled_packs == set()


def test_turn_context_carries_task_state_but_not_enabled_packs():
    from private_agent.tools import begin_turn_tool_context, reset_turn_tool_context

    outer = get_tool_context()
    token = begin_turn_tool_context()
    try:
        turn = get_tool_context()
        turn.task_session_id = "turn-session"
        turn.task_id = "turn-task"
        turn.enabled_packs.add("Coding")
    finally:
        reset_turn_tool_context(token)

    restored = get_tool_context()
    assert restored is outer
    assert restored.task_session_id == "turn-session"
    assert restored.task_id == "turn-task"
    assert restored.task_ids["turn-session"] == "turn-task"
    assert "Coding" not in restored.enabled_packs


@pytest.mark.asyncio
async def test_interleaved_turns_keep_separate_enabled_packs():
    import asyncio

    from private_agent.tools import begin_turn_tool_context, reset_turn_tool_context

    ready = {"first": asyncio.Event(), "second": asyncio.Event()}
    observed = {}

    async def turn(name: str, pack: str, peer: str):
        token = begin_turn_tool_context()
        try:
            get_tool_context().enabled_packs.add(pack)
            ready[name].set()
            await ready[peer].wait()
            first_step = set(get_tool_context().enabled_packs)
            await asyncio.sleep(0)
            get_tool_context().enabled_packs.add(f"{pack}-late")
            observed[name] = (first_step, set(get_tool_context().enabled_packs))
        finally:
            reset_turn_tool_context(token)

    # Both turns are in flight at every step, but their packs never cross over.
    await asyncio.gather(
        asyncio.create_task(turn("first", "Coding", "second")),
        asyncio.create_task(turn("second", "Research", "first")),
    )

    assert observed["first"] == ({"Coding"}, {"Coding", "Coding-late"})
    assert observed["second"] == ({"Research"}, {"Research", "Research-late"})
    assert get_tool_context().enabled_packs == set()


@pytest.mark.asyncio
async def test_concurrent_turns_keep_separate_enabled_packs():
    """Concurrent turns must not share enabled packs or their base context."""
    import asyncio

    from private_agent.tools import begin_turn_tool_context, reset_turn_tool_context

    bases = {}
    observed = {}

    async def turn(name: str, pack: str, peer: str):
        token = begin_turn_tool_context()
        base = get_tool_context()
        bases[name] = id(base)
        try:
            get_tool_context().enabled_packs.add(pack)
            # Simulate enable_capability landing on this turn's context.
            from private_agent.agent import capabilities as _caps

            _caps.enable_capability(pack)
            ready_event = turn_events[name]
            peer_event = turn_events[peer]
            ready_event.set()
            await peer_event.wait()
            observed[name] = set(get_tool_context().enabled_packs)
            await asyncio.sleep(0)
        finally:
            reset_turn_tool_context(token)

    turn_events = {"first": asyncio.Event(), "second": asyncio.Event()}
    await asyncio.gather(
        asyncio.create_task(turn("first", "Coding", "second")),
        asyncio.create_task(turn("second", "Research", "first")),
    )

    assert bases["first"] != bases["second"]
    assert observed["first"] == {"Coding"}
    assert observed["second"] == {"Research"}
    assert get_tool_context().enabled_packs == set()


@pytest.mark.asyncio
async def test_concurrent_turns_do_not_clobber_durable_fields():
    """Durable fields (task_id, ...) must not clobber between concurrent turns."""
    import asyncio

    from private_agent.agent import runtime_f_integration as f
    from private_agent.tools import get_tool_context

    observed = {}

    async def turn(name: str, task_id: str, peer: str):
        state = f.begin_turn(name, "do thing", 100)
        try:
            ctx = get_tool_context()
            ctx.task_session_id = f"sess-{name}"
            ctx.task_id = task_id
            ready = concurrent_events[name]
            peer_ready = concurrent_events[peer]
            ready.set()
            await peer_ready.wait()
            # Each turn must only see its own durable state mid-flight.
            assert get_tool_context().task_id == task_id
            assert get_tool_context().task_session_id == f"sess-{name}"
            observed[name] = (
                get_tool_context().task_id,
                dict(get_tool_context().task_ids),
            )
            await asyncio.sleep(0)
        finally:
            f.finish_turn(state)

    parent_before = get_tool_context()
    parent_task_id_before = parent_before.task_id
    parent_task_ids_before = dict(parent_before.task_ids)

    concurrent_events = {"first": asyncio.Event(), "second": asyncio.Event()}
    await asyncio.gather(
        asyncio.create_task(turn("first", "task-A", "second")),
        asyncio.create_task(turn("second", "task-B", "first")),
    )

    # Neither turn observed the other's task_id mid-flight.
    assert observed["first"][0] == "task-A"
    assert observed["second"][0] == "task-B"
    # The calling (parent) context is not polluted/clobbered by either turn.
    parent = get_tool_context()
    assert parent.task_id == parent_task_id_before
    assert parent.task_ids == parent_task_ids_before


@pytest.mark.asyncio
async def test_worker_thread_tool_execution_sees_the_turn_context():
    import asyncio

    from private_agent.tools import begin_turn_tool_context, reset_turn_tool_context

    token = begin_turn_tool_context()
    try:
        turn = get_tool_context()
        turn.enabled_packs.add("Coding")

        def run_tool_sync():
            # Mirrors the runtime's asyncio.to_thread(tool_func.invoke, args) path.
            context = get_tool_context()
            context.enabled_packs.add("Research")
            context.task_session_id = "thread-session"
            context.task_id = "thread-task"
            set_tool_context(context)
            return id(context)

        assert await asyncio.to_thread(run_tool_sync) == id(turn)

        assert get_tool_context() is turn
        assert get_tool_context().enabled_packs == {"Coding", "Research"}
        assert get_tool_context().task_id == "thread-task"
    finally:
        reset_turn_tool_context(token)
