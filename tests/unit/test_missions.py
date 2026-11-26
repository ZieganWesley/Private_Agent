"""Unit tests for durable mission state and leases."""

import time

import pytest

from database import PersistentMemory


@pytest.fixture
def memory():
    store = PersistentMemory(":memory:")
    try:
        yield store
    finally:
        store.close()


def test_create_and_get_mission(memory):
    mission = memory.create_mission(
        "session-1",
        "Ship the endpoint registry",
        max_iterations=5,
        tool_call_budget=12,
        plan={"steps": [{"step_id": "s1", "description": "add registry"}]},
    )

    assert mission["session_id"] == "session-1"
    assert mission["goal"] == "Ship the endpoint registry"
    assert mission["status"] == "planning"
    assert mission["iteration"] == 0
    assert mission["max_iterations"] == 5
    assert mission["tool_call_budget"] == 12
    assert mission["revision"] == 1
    assert mission["plan"]["steps"][0]["step_id"] == "s1"
    assert memory.get_mission(mission["mission_id"]) == mission


def test_create_mission_rejects_empty_goal(memory):
    with pytest.raises(ValueError):
        memory.create_mission("session-1", "   ")


def test_update_mission_bumps_revision_and_records_event(memory):
    mission = memory.create_mission("session-1", "goal")
    updated = memory.update_mission(
        mission["mission_id"],
        status="active",
        iteration=2,
        tool_calls_used=3,
        progress={"current_step": "s1"},
        expected_revision=mission["revision"],
    )

    assert updated["status"] == "active"
    assert updated["iteration"] == 2
    assert updated["tool_calls_used"] == 3
    assert updated["progress"] == {"current_step": "s1"}
    assert updated["revision"] == mission["revision"] + 1
    events = memory.list_mission_events(mission["mission_id"])
    assert any(event["event_type"] == "updated" for event in events)


def test_update_mission_rejects_stale_revision(memory):
    mission = memory.create_mission("session-1", "goal")
    memory.update_mission(mission["mission_id"], status="active")

    with pytest.raises(ValueError, match="revision"):
        memory.update_mission(
            mission["mission_id"],
            status="paused",
            expected_revision=mission["revision"],
        )


def test_update_mission_rejects_unknown_status(memory):
    mission = memory.create_mission("session-1", "goal")

    with pytest.raises(ValueError):
        memory.update_mission(mission["mission_id"], status="unknown")


def test_list_missions_filters_by_session_and_status(memory):
    first = memory.create_mission("session-a", "one")
    memory.create_mission("session-b", "two")
    memory.update_mission(first["mission_id"], status="active")

    assert [m["goal"] for m in memory.list_missions(session_id="session-a")] == [
        "one"
    ]
    assert [m["goal"] for m in memory.list_missions(status="active")] == ["one"]


def test_mission_lease_is_exclusive_and_releasable(memory):
    mission = memory.create_mission("session-1", "goal")

    assert memory.acquire_mission_lease(mission["mission_id"], "worker-a") is True
    assert memory.mission_lease_active(mission["mission_id"]) is True
    assert memory.acquire_mission_lease(mission["mission_id"], "worker-b") is False
    assert memory.release_mission_lease(mission["mission_id"], "worker-b") is False
    assert memory.release_mission_lease(mission["mission_id"], "worker-a") is True
    assert memory.mission_lease_active(mission["mission_id"]) is False
    assert memory.acquire_mission_lease(mission["mission_id"], "worker-b") is True


def test_mission_lease_expires(memory):
    mission = memory.create_mission("session-1", "goal")
    assert memory.acquire_mission_lease(
        mission["mission_id"], "worker-a", ttl_seconds=0.01
    )
    time.sleep(0.02)

    assert memory.mission_lease_active(mission["mission_id"]) is False
    assert memory.acquire_mission_lease(mission["mission_id"], "worker-b") is True
