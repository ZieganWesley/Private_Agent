"""Tests for the snapshot manifest and turn-scoped undo (roadmap 6.7)."""

import pathlib

import pytest

from code_tasks import snapshots
from sandbox import SandboxManager, create_hitl_snapshot
from tools import (
    delete_workspace_file,
    edit_local_file,
    get_tool_context,
    rename_workspace_file,
    set_tool_context,
)


@pytest.fixture(autouse=True)
def restore_turn_context():
    """Keep the shared tool context clean between tests."""
    context = get_tool_context()
    original = (context.session_id, context.turn_id)
    yield context
    context.session_id, context.turn_id = original
    set_tool_context(context)


def _use_turn(session_id="session-1", turn_id="turn-1"):
    context = get_tool_context()
    context.session_id = session_id
    context.turn_id = turn_id
    set_tool_context(context)


def _backup(root: pathlib.Path, name: str, content: str) -> str:
    directory = root / snapshots.SNAPSHOT_DIR_NAME
    directory.mkdir(exist_ok=True)
    backup_name = f"{name}_{len(list(directory.iterdir()))}.bak"
    (directory / backup_name).write_text(content, encoding="utf-8")
    return backup_name


def _record_modify(root, target, content, *, session_id="session-1", turn_id="turn-1"):
    backup = _backup(root, pathlib.Path(target).name, content)
    return snapshots.record_snapshot(
        root=root,
        target=target,
        backup=backup,
        session_id=session_id,
        turn_id=turn_id,
    )


def _record_create(root, target, *, session_id="session-1", turn_id="turn-1"):
    return snapshots.record_snapshot(
        root=root,
        target=target,
        action="create",
        session_id=session_id,
        turn_id=turn_id,
    )


def test_record_snapshot_appends_a_manifest_entry(temp_workspace):
    record = _record_modify(temp_workspace, "notes.txt", "one")

    stored = snapshots.read_records(temp_workspace)

    assert len(stored) == 1
    assert stored[0].action == "modify"
    assert stored[0].target == "notes.txt"
    assert stored[0].backup == record.backup
    assert stored[0].session_id == "session-1"
    assert stored[0].turn_id == "turn-1"
    assert stored[0].size == 3
    assert len(stored[0].sha256) == 64
    assert stored[0].created_at


def test_record_snapshot_requires_a_session_and_turn(temp_workspace):
    with pytest.raises(snapshots.SnapshotError):
        snapshots.record_snapshot(
            root=temp_workspace,
            target="notes.txt",
            action="create",
            session_id=None,
            turn_id="turn-1",
        )
    with pytest.raises(snapshots.SnapshotError):
        snapshots.record_snapshot(
            root=temp_workspace,
            target="notes.txt",
            action="create",
            session_id="session-1",
            turn_id=None,
        )


def test_record_snapshot_rejects_paths_outside_the_workspace(
    temp_workspace, tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(snapshots.SnapshotError):
        snapshots.record_snapshot(
            root=temp_workspace,
            target=outside,
            action="create",
            session_id="session-1",
            turn_id="turn-1",
        )


def test_read_records_skips_damaged_and_future_lines(temp_workspace):
    _record_modify(temp_workspace, "notes.txt", "one")
    manifest = snapshots.manifest_path(temp_workspace)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write('{"version": 99, "action": "modify"}\n')
        handle.write("\n")
    _record_create(temp_workspace, "new.txt")

    stored = snapshots.read_records(temp_workspace)

    assert [record.target for record in stored] == ["notes.txt", "new.txt"]


def test_read_records_without_a_manifest_returns_empty(temp_workspace):
    assert snapshots.read_records(temp_workspace) == []


def test_undo_restores_modified_files_and_removes_created_ones(temp_workspace):
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "pre-turn")
    target.write_text("post-turn", encoding="utf-8")
    created = temp_workspace / "created.txt"
    created.write_text("new", encoding="utf-8")
    _record_create(temp_workspace, "created.txt")

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "undone"
    assert result.turn_id == "turn-1"
    assert result.restored == ("notes.txt",)
    assert result.deleted == ("created.txt",)
    assert target.read_text(encoding="utf-8") == "pre-turn"
    assert not created.exists()


def test_undo_uses_the_first_snapshot_of_a_file_changed_twice(temp_workspace):
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "pre-turn")
    target.write_text("mid-turn", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "mid-turn")
    target.write_text("post-turn", encoding="utf-8")

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "undone"
    assert result.restored == ("notes.txt",)
    assert target.read_text(encoding="utf-8") == "pre-turn"


def test_undo_walks_back_one_turn_at_a_time(temp_workspace):
    target = temp_workspace / "notes.txt"
    target.write_text("turn-1", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "turn-1", turn_id="turn-1")
    target.write_text("turn-2", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "turn-2", turn_id="turn-2")
    target.write_text("turn-3", encoding="utf-8")

    first = snapshots.undo_turn(root=temp_workspace, session_id="session-1")
    second = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert first.turn_id == "turn-2"
    assert second.turn_id == "turn-1"
    assert target.read_text(encoding="utf-8") == "turn-1"


def test_undo_reports_already_undone_turns(temp_workspace):
    _record_create(temp_workspace, "created.txt")

    first = snapshots.undo_turn(root=temp_workspace, session_id="session-1")
    second = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert first.status == "undone"
    assert second.status == "empty"
    explicit = snapshots.undo_turn(
        root=temp_workspace,
        session_id="session-1",
        turn_id="turn-1",
    )
    assert explicit.status == "already_undone"


def test_undo_returns_empty_without_records(temp_workspace):
    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "empty"
    assert result.turn_id is None


def test_undo_ignores_other_sessions(temp_workspace):
    target = temp_workspace / "notes.txt"
    target.write_text("other", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "other", session_id="other-session")

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "empty"
    assert target.read_text(encoding="utf-8") == "other"


def test_undo_reports_a_missing_backup_without_touching_the_file(temp_workspace):
    record = _record_modify(temp_workspace, "notes.txt", "pre-turn")
    target = temp_workspace / "notes.txt"
    target.write_text("post-turn", encoding="utf-8")
    (temp_workspace / snapshots.SNAPSHOT_DIR_NAME / record.backup).unlink()

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "partial"
    assert "no longer exists" in result.errors[0]
    assert target.read_text(encoding="utf-8") == "post-turn"
    # Not marked undone, so a retry after fixing the backup is still possible.
    assert snapshots.undoable_turn_ids(
        snapshots.read_records(temp_workspace), "session-1"
    ) == ["turn-1"]


def test_undo_reports_a_backup_that_changed_after_recording(temp_workspace):
    record = _record_modify(temp_workspace, "notes.txt", "pre-turn")
    target = temp_workspace / "notes.txt"
    target.write_text("post-turn", encoding="utf-8")
    (temp_workspace / snapshots.SNAPSHOT_DIR_NAME / record.backup).write_text(
        "tampered", encoding="utf-8"
    )

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "partial"
    assert "changed since it was recorded" in result.errors[0]
    assert target.read_text(encoding="utf-8") == "post-turn"


def test_undo_leaves_the_file_intact_when_the_restore_cannot_be_written(
    temp_workspace, monkeypatch
):
    _record_modify(temp_workspace, "notes.txt", "pre-turn")
    target = temp_workspace / "notes.txt"
    target.write_text("post-turn", encoding="utf-8")

    def refuse(source, destination):
        raise PermissionError("restore target is not writable")

    monkeypatch.setattr(snapshots.os, "replace", refuse)

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "partial"
    assert "not writable" in result.errors[0]
    assert target.read_text(encoding="utf-8") == "post-turn"
    assert not list(temp_workspace.glob(".undo-*"))


def test_undo_refuses_to_restore_through_a_symlink(temp_workspace, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "target.txt"
    outside.write_text("outside", encoding="utf-8")
    _record_modify(temp_workspace, "notes.txt", "pre-turn")
    link = temp_workspace / "notes.txt"
    link.symlink_to(outside)

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "partial"
    assert "symlink" in result.errors[0]
    assert outside.read_text(encoding="utf-8") == "outside"


def test_undo_refuses_to_delete_a_created_path_that_is_no_longer_a_file(
    temp_workspace,
):
    created = temp_workspace / "created.txt"
    created.mkdir()
    _record_create(temp_workspace, "created.txt")

    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert result.status == "partial"
    assert created.is_dir()


def test_create_hitl_snapshot_records_the_turn_and_undo_restores_it(
    temp_workspace,
):
    _use_turn()
    target = temp_workspace / "important.txt"
    target.write_text("pre-turn", encoding="utf-8")

    backup_path = create_hitl_snapshot(target)
    target.write_text("post-turn", encoding="utf-8")

    assert backup_path is not None and backup_path.exists()
    records = snapshots.read_records(temp_workspace)
    assert [(record.action, record.target) for record in records] == [
        ("modify", "important.txt")
    ]
    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")
    assert result.status == "undone"
    assert target.read_text(encoding="utf-8") == "pre-turn"


def test_hitl_snapshot_outside_a_turn_stays_unrecorded(temp_workspace):
    _use_turn(turn_id=None)
    target = temp_workspace / "important.txt"
    target.write_text("content", encoding="utf-8")

    assert create_hitl_snapshot(target) is not None
    assert snapshots.read_records(temp_workspace) == []


def test_write_tool_edit_is_undoable(temp_workspace):
    _use_turn()
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")

    edited = edit_local_file.invoke({"file_path": "notes.txt", "content": "post-turn"})
    records = snapshots.read_records(temp_workspace)
    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert "Success" in edited
    assert [(record.action, record.target) for record in records] == [
        ("modify", "notes.txt")
    ]
    assert result.status == "undone"
    assert target.read_text(encoding="utf-8") == "pre-turn"


def test_write_tool_creation_is_removed_by_undo(temp_workspace):
    _use_turn()
    created = temp_workspace / "new.txt"

    edit_local_file.invoke({"file_path": "new.txt", "content": "fresh"})
    records = snapshots.read_records(temp_workspace)
    result = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert [(record.action, record.target) for record in records] == [
        ("create", "new.txt")
    ]
    assert result.status == "undone"
    assert result.deleted == ("new.txt",)
    assert not created.exists()


def test_rename_and_delete_tools_are_undoable(temp_workspace):
    _use_turn()
    source = temp_workspace / "source.txt"
    source.write_text("body", encoding="utf-8")

    rename_workspace_file.invoke(
        {"source_path": "source.txt", "destination_path": "moved.txt"}
    )
    undo_rename = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert undo_rename.status == "undone"
    assert source.read_text(encoding="utf-8") == "body"
    assert not (temp_workspace / "moved.txt").exists()

    _use_turn(turn_id="turn-2")
    delete_workspace_file.invoke({"file_path": "source.txt"})
    undo_delete = snapshots.undo_turn(root=temp_workspace, session_id="session-1")

    assert undo_delete.status == "undone"
    assert source.read_text(encoding="utf-8") == "body"


def test_tool_edits_outside_a_turn_are_not_recorded(temp_workspace):
    _use_turn(turn_id=None)
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")

    edit_local_file.invoke({"file_path": "notes.txt", "content": "post-turn"})

    assert snapshots.read_records(temp_workspace) == []
    assert target.read_text(encoding="utf-8") == "post-turn"
    assert SandboxManager.get_root() == temp_workspace.resolve()


def test_manifest_write_failure_does_not_block_the_edit(temp_workspace, monkeypatch):
    _use_turn()
    target = temp_workspace / "notes.txt"
    target.write_text("pre-turn", encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(snapshots, "_append_record", boom)

    result = edit_local_file.invoke({"file_path": "notes.txt", "content": "post-turn"})

    assert "Success" in result
    assert target.read_text(encoding="utf-8") == "post-turn"
