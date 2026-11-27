"""Unit tests for /mission command routing."""

import pytest

from agent.commands.mission import handle_mission_command
from database import PersistentMemory


class FakeUI:
    def __init__(self):
        self.messages = []

    def print(self, text):
        self.messages.append(text)


@pytest.fixture
def memory():
    store = PersistentMemory(":memory:")
    try:
        yield store
    finally:
        store.close()


def test_empty_argument_prints_usage(memory):
    ui = FakeUI()

    result = handle_mission_command(
        memory=memory, session_id="s", argument="", session_ui=ui
    )

    assert result is None
    assert "Usage:" in ui.messages[0]


def test_start_creates_a_mission(memory):
    ui = FakeUI()

    result = handle_mission_command(
        memory=memory,
        session_id="s",
        argument='start "ship the registry"',
        session_ui=ui,
    )

    assert result is not None
    controller = result["controller"]
    mission = controller.mission()
    assert mission["goal"] == "ship the registry"
    assert [step["step_id"] for step in mission["plan"]["steps"]] == [
        "step-1",
        "step-2",
        "step-3",
    ]
    assert "Mission started" in ui.messages[0]


def test_start_requires_a_goal(memory):
    ui = FakeUI()

    assert (
        handle_mission_command(
            memory=memory, session_id="s", argument="start", session_ui=ui
        )
        is None
    )
    assert "goal is required" in ui.messages[0]


def test_list_and_status(memory):
    started = handle_mission_command(
        memory=memory,
        session_id="s",
        argument="start goal",
        session_ui=FakeUI(),
    )
    mission_id = started["controller"].mission_id
    ui = FakeUI()

    listed = handle_mission_command(
        memory=memory, session_id="s", argument="list", session_ui=ui
    )
    assert [m["mission_id"] for m in listed["missions"]] == [mission_id]
    assert mission_id in ui.messages[0]

    ui.messages.clear()
    status = handle_mission_command(
        memory=memory, session_id="s", argument=f"status {mission_id}", session_ui=ui
    )
    assert status["mission"]["mission_id"] == mission_id


def test_pause_resume_stop(memory):
    started = handle_mission_command(
        memory=memory,
        session_id="s",
        argument="start goal",
        session_ui=FakeUI(),
    )
    mission_id = started["controller"].mission_id
    ui = FakeUI()

    paused = handle_mission_command(
        memory=memory, session_id="s", argument=f"pause {mission_id}", session_ui=ui
    )
    assert paused["mission"]["status"] == "paused"

    resumed = handle_mission_command(
        memory=memory, session_id="s", argument=f"resume {mission_id}", session_ui=ui
    )
    assert resumed["mission"]["status"] == "active"

    stopped = handle_mission_command(
        memory=memory, session_id="s", argument=f"stop {mission_id}", session_ui=ui
    )
    assert stopped["mission"]["status"] == "stopped"


def test_prompt_control_pauses_and_resumes_the_active_mission(memory, monkeypatch):
    from agent.commands import mission_runtime
    from agent.mission_runner import MissionAutoRunner

    started = handle_mission_command(
        memory=memory,
        session_id="s",
        argument="start goal",
        session_ui=FakeUI(),
    )
    runner = MissionAutoRunner(started["controller"], "owner-1")
    assert runner.next_prompt() is not None
    monkeypatch.setattr(mission_runtime, "_ACTIVE_RUNNER", runner)
    ui = FakeUI()
    try:
        assert mission_runtime.mission_slash_command("/pause", memory, "s", ui) is None
        assert runner.controller.mission()["status"] == "paused"

        resume_prompt = mission_runtime.mission_slash_command(
            "/resume", memory, "s", ui
        )

        assert resume_prompt is not None and "step-1" in resume_prompt
        assert runner.controller.mission()["status"] == "active"
        assert any("Mission paused" in message for message in ui.messages)
    finally:
        runner.controller.release_lease(runner.owner)


def test_mission_start_announces_live_controls(memory, monkeypatch):
    from agent.commands import mission_runtime

    monkeypatch.setattr(mission_runtime, "_ACTIVE_RUNNER", None)
    ui = FakeUI()

    prompt = mission_runtime.mission_slash_command(
        "/mission start goal", memory, "s", ui
    )

    assert prompt is not None and "step-1" in prompt
    assert any("Mission controls" in message for message in ui.messages)
    assert any("/pause" in message and "/stop" in message for message in ui.messages)
    runner = mission_runtime._ACTIVE_RUNNER
    if runner is not None:
        runner.controller.release_lease(runner.owner)
    monkeypatch.setattr(mission_runtime, "_ACTIVE_RUNNER", None)


def test_unknown_action(memory):
    ui = FakeUI()

    assert (
        handle_mission_command(
            memory=memory, session_id="s", argument="nope", session_ui=ui
        )
        is None
    )
    assert "Unknown action" in ui.messages[0]
