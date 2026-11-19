"""Tests for the `/undo` command outcome and its failure messages (6.7)."""

import pathlib
import sqlite3

import pytest

from private_agent import snapshots
from private_agent.agent.undo import undo_last_turn
from private_agent.database import PersistentMemory
from private_agent.tools import edit_local_file, get_tool_context


@pytest.fixture(autouse=True)
def restore_turn_context():
    context = get_tool_context()
    original = (context.session_id, context.turn_id)
    yield context
    context.session_id, context.turn_id = original


@pytest.fixture
def memory(temp_workspace):
    database = PersistentMemory(str(temp_workspace / "undo.sqlite"))
    yield database
    database.close()


@pytest.fixture
def turn(memory):
    context = get_tool_context()
    context.session_id = "session-1"
    context.turn_id = "turn-1"
    return context


def _save_turn(memory, turn_id="turn-1", session_id="session-1"):
    memory.save_message(session_id, "human", "edit notes", turn_id=turn_id)
    memory.save_message(session_id, "ai", "edited notes", turn_id=turn_id)


def test_undo_restores_the_file_and_rewinds_the_conversation(
    memory, temp_workspace, turn
):
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")
    edit_local_file.invoke({"file_path": "notes.txt", "content": "post-turn"})
    _save_turn(memory)

    outcome = undo_last_turn(
        memory,
        "session-1",
        workspace_root=temp_workspace,
    )

    assert outcome.status == "undone"
    assert outcome.history_changed is True
    assert outcome.restored == ("notes.txt",)
    assert target.read_text(encoding="utf-8") == "pre-turn"
    assert memory.get_history("session-1") == []
    assert "Restored 1 file(s)" in outcome.message
    assert "removed 2 message(s)" in outcome.message


def test_undo_removes_a_file_the_turn_created(memory, temp_workspace, turn):
    created = temp_workspace / "created.txt"
    edit_local_file.invoke({"file_path": "created.txt", "content": "fresh"})
    _save_turn(memory)

    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "undone"
    assert outcome.deleted == ("created.txt",)
    assert not created.exists()
    assert "Removed 1 created file(s)" in outcome.message


def test_undo_reports_nothing_to_undo(memory, temp_workspace):
    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "empty"
    assert outcome.history_changed is False
    assert "Nothing to undo" in outcome.message


def test_undo_reports_an_already_undone_turn(memory, temp_workspace, turn):
    edit_local_file.invoke({"file_path": "created.txt", "content": "fresh"})
    _save_turn(memory)

    undo_last_turn(memory, "session-1", workspace_root=temp_workspace)
    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "empty"
    assert "already" not in outcome.message.lower()
    explicit = undo_last_turn(
        memory,
        "session-1",
        workspace_root=temp_workspace,
        turn_id="turn-1",
    )
    assert explicit.status == "already_undone"
    assert "already undone" in explicit.message


def test_undo_keeps_the_conversation_when_a_file_cannot_be_restored(
    memory, temp_workspace, turn
):
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")
    edit_local_file.invoke({"file_path": "notes.txt", "content": "post-turn"})
    _save_turn(memory)
    backup = snapshots.read_records(temp_workspace)[0].backup
    (temp_workspace / snapshots.SNAPSHOT_DIR_NAME / backup).unlink()

    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "partial"
    assert outcome.history_changed is False
    assert target.read_text(encoding="utf-8") == "post-turn"
    assert len(memory.get_history("session-1")) == 2
    assert "Could not fully restore" in outcome.message
    assert "left untouched" in outcome.message


def test_undo_reports_a_history_failure_after_restoring_files(
    memory, temp_workspace, turn, monkeypatch
):
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")
    edit_local_file.invoke({"file_path": "notes.txt", "content": "post-turn"})
    _save_turn(memory)

    def refuse(session_id, turn_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(memory, "delete_turn_messages", refuse)

    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "partial"
    assert outcome.history_changed is False
    assert target.read_text(encoding="utf-8") == "pre-turn"
    assert "already back" in outcome.message


def test_undo_reports_an_unreadable_manifest(memory, temp_workspace):
    manifest = snapshots.manifest_path(temp_workspace)
    manifest.mkdir(parents=True)

    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "error"
    assert "Nothing was changed" in outcome.message
    assert outcome.history_changed is False


def test_undo_keeps_other_turns_and_summaries(memory, temp_workspace, turn):
    target = temp_workspace / "notes.txt"
    target.write_text("turn-1", encoding="utf-8")
    edit_local_file.invoke({"file_path": "notes.txt", "content": "turn-1-edited"})
    _save_turn(memory, turn_id="turn-1")

    context = get_tool_context()
    context.turn_id = "turn-2"
    edit_local_file.invoke({"file_path": "notes.txt", "content": "turn-2-edited"})
    _save_turn(memory, turn_id="turn-2")
    memory.save_message("session-1", "summary", "older conversation")

    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "undone"
    assert target.read_text(encoding="utf-8") == "turn-1-edited"
    rows = memory.conn.execute(
        "SELECT role, turn_id FROM chat_history WHERE session_id = ? ORDER BY id",
        ("session-1",),
    ).fetchall()
    assert rows == [("human", "turn-1"), ("ai", "turn-1"), ("summary", None)]


def test_undo_survives_a_workspace_path_argument_as_a_string(
    memory, temp_workspace, turn
):
    edit_local_file.invoke({"file_path": "created.txt", "content": "fresh"})
    _save_turn(memory)

    outcome = undo_last_turn(
        memory,
        "session-1",
        workspace_root=str(pathlib.Path(temp_workspace)),
    )

    assert outcome.status == "undone"


def test_undo_message_escapes_markup_in_file_names(memory, temp_workspace, turn):
    target = temp_workspace / "a[b].txt"
    target.write_text("pre-turn", encoding="utf-8")
    edit_local_file.invoke({"file_path": "a[b].txt", "content": "post-turn"})
    _save_turn(memory)

    outcome = undo_last_turn(memory, "session-1", workspace_root=temp_workspace)

    assert outcome.status == "undone"
    assert "a\\[b].txt" in outcome.message
    assert target.read_text(encoding="utf-8") == "pre-turn"
