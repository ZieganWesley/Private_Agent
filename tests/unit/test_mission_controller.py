"""Unit tests for the mission state machine and guards."""

import time

import pytest

from agent.missions import MissionController
from database import PersistentMemory


@pytest.fixture
def memory():
    store = PersistentMemory(":memory:")
    try:
        yield store
    finally:
        store.close()


def _controller(memory, **kwargs):
    return MissionController.start(
        memory,
        "session-1",
        "ship the endpoint registry",
        [
            {"step_id": "s1", "description": "add registry", "acceptance": ["tests"]},
            {"step_id": "s2", "description": "wire discovery"},
        ],
        **kwargs,
    )


def test_start_persists_plan_and_planning_status(memory):
    controller = _controller(memory)
    mission = controller.mission()

    assert mission["status"] == "planning"
    assert [step["step_id"] for step in mission["plan"]["steps"]] == ["s1", "s2"]
    assert mission["plan"]["steps"][0]["status"] == "pending"


def test_begin_iteration_activates_and_counts(memory):
    controller = _controller(memory, max_iterations=2)

    assert controller.begin_iteration() is True
    assert controller.mission()["status"] == "active"
    assert controller.mission()["iteration"] == 1


def test_begin_iteration_fails_at_iteration_budget(memory):
    controller = _controller(memory, max_iterations=1)
    assert controller.begin_iteration() is True

    assert controller.begin_iteration() is False
    assert controller.mission()["status"] == "failed"


def test_begin_iteration_fails_after_deadline(memory):
    controller = _controller(memory, deadline=time.time() - 1)

    assert controller.begin_iteration() is False
    assert controller.mission()["status"] == "failed"


def test_next_step_marks_first_pending_step_active(memory):
    controller = _controller(memory)
    controller.begin_iteration()

    step = controller.next_step()

    assert step["step_id"] == "s1"
    assert controller.mission()["plan"]["steps"][0]["status"] == "active"


def test_complete_step_requires_evidence(memory):
    controller = _controller(memory)
    controller.begin_iteration()
    controller.next_step()

    with pytest.raises(ValueError, match="evidence"):
        controller.complete_step("s1", "   ")


def test_complete_mission_requires_every_step(memory):
    controller = _controller(memory)
    controller.begin_iteration()
    controller.next_step()
    controller.complete_step("s1", "tests passed")

    with pytest.raises(ValueError, match="incomplete"):
        controller.complete_mission("done", "all tests passed")


def test_complete_mission_after_all_steps(memory):
    controller = _controller(memory)
    controller.begin_iteration()
    controller.next_step()
    controller.complete_step("s1", "tests passed")
    controller.next_step()
    controller.complete_step("s2", "discovery works")

    mission = controller.complete_mission("done", "all tests passed")

    assert mission["status"] == "completed"
    assert mission["completion_evidence"] == "all tests passed"


def test_tool_budget_guard_fails_mission(memory):
    controller = _controller(memory, tool_call_budget=1)
    controller.begin_iteration()

    assert (
        controller.record_tool_call("read_file", {"path": "a"}, success=True) is False
    )
    assert controller.mission()["status"] == "failed"


def test_repeated_identical_tool_call_trips_no_progress_guard(memory):
    controller = _controller(memory)
    controller.begin_iteration()

    assert controller.record_tool_call("read_file", {"path": "a"}, success=True)
    assert controller.record_tool_call("read_file", {"path": "a"}, success=True)
    assert (
        controller.record_tool_call("read_file", {"path": "a"}, success=True) is False
    )
    assert controller.mission()["status"] == "failed"
    assert "No progress" in controller.mission()["last_error"]


def test_consecutive_failures_fail_mission(memory):
    controller = _controller(memory)
    controller.begin_iteration()

    assert controller.record_tool_call("shell", {}, success=False)
    assert controller.record_tool_call("shell", {"n": 2}, success=False)
    assert controller.record_tool_call("shell", {"n": 3}, success=False) is False
    assert "consecutive" in controller.mission()["last_error"]


def test_pause_resume_and_stop(memory):
    controller = _controller(memory)
    controller.begin_iteration()

    assert controller.pause()["status"] == "paused"
    assert controller.resume()["status"] == "active"
    assert controller.stop()["status"] == "stopped"
    assert controller.begin_iteration() is False


def test_events_are_appended(memory):
    controller = _controller(memory)
    controller.begin_iteration()
    controller.next_step()
    controller.complete_step("s1", "proof")

    events = [
        event["event_type"]
        for event in memory.list_mission_events(controller.mission_id)
    ]
    assert "planned" in events
    assert "iteration_started" in events
    assert "step_started" in events
    assert "step_completed" in events
