"""Focused tests for automatic mission iterations and the controller lease."""

import pytest

from agent.mission_runner import MissionAutoRunner
from agent.missions import MissionController, MissionLeaseError
from database import PersistentMemory


@pytest.fixture
def memory():
    store = PersistentMemory(":memory:")
    try:
        yield store
    finally:
        store.close()


def _controller(memory):
    return MissionController.start(
        memory,
        "session-1",
        "ship the endpoint registry",
        [
            {"step_id": "s1", "description": "add registry", "acceptance": ["tests"]},
            {"step_id": "s2", "description": "wire discovery"},
        ],
    )


def test_runner_advances_multiple_iterations_from_model_evidence(memory):
    controller = _controller(memory)
    runner = MissionAutoRunner(controller, "owner-1")

    first = runner.next_prompt()
    assert first is not None and "s1" in first
    assert controller.mission()["iteration"] == 1

    second = runner.observe("registry tests passed", had_tool_calls=False)
    assert second is not None and "s2" in second
    assert controller.mission()["iteration"] == 2

    assert runner.observe("discovery verified", had_tool_calls=False) is None
    mission = controller.mission()
    assert mission["status"] == "completed"
    assert mission["completion_evidence"] == "discovery verified"
    assert [step["status"] for step in mission["plan"]["steps"]] == [
        "completed",
        "completed",
    ]

    events = [
        event["event_type"]
        for event in memory.list_mission_events(mission["mission_id"])
    ]
    assert events.count("iteration_started") == 2
    assert events.count("step_completed") == 2
    assert "completed" in events


def test_tool_call_turn_does_not_complete_the_step(memory):
    controller = _controller(memory)
    runner = MissionAutoRunner(controller, "owner-1")
    runner.next_prompt()

    prompt = runner.observe("still working", had_tool_calls=True)

    assert prompt is not None and "s1" in prompt
    assert controller.mission()["plan"]["steps"][0]["status"] == "active"
    assert controller.mission()["iteration"] == 1


def test_interrupted_turn_is_not_completion_evidence(memory):
    controller = _controller(memory)
    runner = MissionAutoRunner(controller, "owner-1")
    runner.next_prompt()

    prompt = runner.observe("partial", had_tool_calls=False, interrupted=True)

    assert prompt is not None
    assert controller.mission()["plan"]["steps"][0]["status"] == "active"


def test_pause_keeps_the_lease_and_resume_returns_the_current_step(memory):
    controller = _controller(memory)
    runner = MissionAutoRunner(controller, "owner-1")
    assert runner.next_prompt() is not None

    controller.pause()
    assert runner.observe("partial", had_tool_calls=False) is None
    assert memory.mission_lease_active(controller.mission_id) is True

    controller.resume()
    prompt = runner.resume_prompt()

    assert prompt is not None and "s1" in prompt
    assert controller.mission()["status"] == "active"


def test_stopped_mission_stops_prompting_and_releases_its_lease(memory):
    controller = _controller(memory)
    runner = MissionAutoRunner(controller, "owner-1")
    assert runner.next_prompt() is not None

    controller.stop("user cancelled")

    assert runner.observe("stopped", had_tool_calls=False) is None
    assert memory.mission_lease_active(controller.mission_id) is False


def test_lease_blocks_a_second_controller(memory):
    controller = _controller(memory)
    first = MissionAutoRunner(controller, "owner-1")

    with pytest.raises(MissionLeaseError):
        MissionAutoRunner(controller, "owner-2")

    first.controller.release_lease("owner-1")
    second = MissionAutoRunner(controller, "owner-2")
    assert second.next_prompt() is not None
