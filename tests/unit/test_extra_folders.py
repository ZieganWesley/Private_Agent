from pathlib import Path

from private_agent.sandbox import SandboxManager


def test_approved_extra_folder_allows_read(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "note.txt").write_text("hello", encoding="utf-8")

    SandboxManager.set_root(str(workspace))
    SandboxManager.set_approved_extra_folders([str(extra)])

    target = SandboxManager.validate_path(str(extra / "note.txt"))
    assert target == (extra / "note.txt").resolve()

    # Outside both workspace and approved folders is still denied.
    other = tmp_path / "other"
    other.mkdir()
    try:
        SandboxManager.validate_path(str(other / "x.txt"))
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError")
