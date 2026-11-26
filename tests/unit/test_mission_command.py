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


def test_unknown_action(memory):
    ui = FakeUI()

    assert (
        handle_mission_command(
            memory=memory, session_id="s", argument="nope", session_ui=ui
        )
        is None
    )
    assert "Unknown action" in ui.messages[0]
