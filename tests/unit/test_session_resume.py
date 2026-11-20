"""Session resume: id resolution, history helpers and task-lease takeover."""

import sqlite3

import pytest

from private_agent.agent.session_resume import (
    SessionResumeError,
    resolve_session_id,
    take_over_session_task_lease,
)
from private_agent.database import PersistentMemory


@pytest.fixture
def memory(tmp_path):
    store = PersistentMemory(str(tmp_path / "sessions.sqlite"))
    yield store
    store.close()


def _active_task(memory, session_id: str = "session_a") -> dict:
    task = memory.create_task_plan(
        session_id,
        "Resume a saved session",
        "coding",
        [{"step_id": "inspect", "description": "Inspect the saved state"}],
    )
    return memory.approve_task_plan(
        task["task_id"], task["revision"], task["digest"], "coding: resume test"
    )


def test_resume_flag_returns_the_saved_session(memory):
    memory.save_message("session_a", "human", "hello")

    assert resolve_session_id(memory, resume_session=" session_a ") == "session_a"


def test_unknown_resume_id_raises_a_startup_error(memory):
    with pytest.raises(SessionResumeError) as excinfo:
        resolve_session_id(memory, resume_session="session_missing")

    message = str(excinfo.value)
    assert "session_missing" in message
    assert "--list-sessions" in message
    assert isinstance(excinfo.value, RuntimeError)


def test_continue_without_history_raises(memory):
    with pytest.raises(SessionResumeError):
        resolve_session_id(memory, continue_session=True)


def test_continue_picks_the_most_recent_session(memory):
    memory.save_message("session_old", "human", "first")
    memory.save_message("session_new", "human", "second")

    assert resolve_session_id(memory, continue_session=True) == "session_new"


def test_explicit_resume_wins_over_the_interactive_prompt(memory):
    memory.save_message("session_a", "human", "hello")
    asked = []

    resolved = resolve_session_id(
        memory,
        resume_session="session_a",
        prompt=lambda latest: asked.append(latest) or "n",
    )

    assert resolved == "session_a"
    assert asked == []


def test_interactive_prompt_starts_a_new_session_when_declined(memory):
    memory.save_message("session_a", "human", "hello")

    resolved = resolve_session_id(memory, prompt=lambda _latest: "n")

    assert resolved != "session_a"
    assert resolved.startswith("session_")


def test_interactive_prompt_resumes_on_empty_answer(memory):
    memory.save_message("session_a", "human", "hello")

    assert resolve_session_id(memory, prompt=lambda _latest: "") == "session_a"


def test_prompt_is_skipped_without_saved_sessions(memory):
    asked = []

    resolved = resolve_session_id(
        memory, prompt=lambda latest: asked.append(latest) or "y"
    )

    assert asked == []
    assert resolved.startswith("session_")


def test_session_exists_only_matches_saved_rows(memory):
    memory.save_message("session_a", "human", "hello")

    assert memory.session_exists("session_a") is True
    assert memory.session_exists("session_b") is False
    assert memory.session_exists("") is False


def test_ends_with_summary_tracks_new_activity(memory):
    memory.save_message("session_a", "human", "hello")
    assert memory.ends_with_summary("session_a") is False

    memory.save_summary("session_a", "Did the thing.")
    assert memory.ends_with_summary("session_a") is True

    memory.save_message("session_a", "ai", "more work")
    assert memory.ends_with_summary("session_a") is False


def test_list_sessions_with_metadata_orders_by_activity(memory):
    memory.save_message("session_old", "human", "first")
    memory.save_message("session_new", "human", "second")
    memory.save_message("session_new", "ai", "second reply")
    memory.save_summary("session_new", "Summary of the new session.")

    sessions = memory.list_sessions_with_metadata()

    assert [session["session_id"] for session in sessions] == [
        "session_new",
        "session_old",
    ]
    assert sessions[0]["messages"] == 2
    assert sessions[1]["messages"] == 1


def test_startup_episodes_can_exclude_the_resumed_session(memory):
    memory.save_summary("session_a", "Episode from the resumed session.")
    memory.save_summary("session_b", "Episode from another session.")

    excluded = memory.get_recent_episodic_memories(5, exclude_session_id="session_a")

    assert [episode["session_id"] for episode in excluded] == ["session_b"]
    assert len(memory.get_recent_episodic_memories(5)) == 2


def test_session_active_task_returns_only_active_plans(memory):
    assert memory.get_session_active_task("session_a") is None

    task = _active_task(memory)

    assert memory.get_session_active_task("session_a")["task_id"] == task["task_id"]
    memory.set_task_status(task["task_id"], "paused")
    assert memory.get_session_active_task("session_a") is None


def test_malformed_active_task_is_ignored(memory, monkeypatch):
    _active_task(memory)
    monkeypatch.setattr(
        memory,
        "get_task",
        lambda _task_id: (_ for _ in ()).throw(ValueError("malformed plan")),
    )

    assert memory.get_session_active_task("session_a") is None


def test_lease_takeover_claims_a_free_active_task(memory):
    task = _active_task(memory)

    status = take_over_session_task_lease(memory, "session_a", "owner_new")

    assert status == "acquired"
    lease = memory.get_task_lease(task["task_id"])
    assert lease["owner_id"] == "owner_new"
    assert lease["session_id"] == "session_a"


def test_lease_takeover_takes_an_expired_foreign_lease(memory):
    task = _active_task(memory)
    memory.acquire_task_lease(task["task_id"], "owner_dead", "session_a")
    memory.conn.execute(
        "UPDATE agent_task_leases SET expires_at = 0 WHERE task_id = ?",
        (task["task_id"],),
    )
    memory.conn.commit()

    status = take_over_session_task_lease(memory, "session_a", "owner_new")

    assert status == "acquired"
    assert memory.get_task_lease(task["task_id"])["owner_id"] == "owner_new"


def test_lease_takeover_never_steals_a_live_lease(memory):
    task = _active_task(memory)
    memory.acquire_task_lease(task["task_id"], "owner_running", "session_a")

    status = take_over_session_task_lease(memory, "session_a", "owner_new")

    assert status == "held"
    assert memory.get_task_lease(task["task_id"])["owner_id"] == "owner_running"


def test_lease_takeover_reports_database_failures(memory, monkeypatch):
    _active_task(memory)

    def _broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(memory, "acquire_task_lease", _broken)

    assert take_over_session_task_lease(memory, "session_a", "owner_new") == "failed"


def test_lease_takeover_skips_sessions_without_an_active_plan(memory):
    assert take_over_session_task_lease(memory, "session_a", "owner_new") is None


def test_lease_takeover_reports_task_lookup_failures(memory, monkeypatch):
    def _broken(_session_id):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(memory, "get_session_active_task", _broken)

    assert take_over_session_task_lease(memory, "session_a", "owner_new") == "failed"
