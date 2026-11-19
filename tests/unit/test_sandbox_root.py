import pytest

from private_agent.sandbox import SandboxManager


def test_workspace_root_requires_explicit_configuration(monkeypatch):
    monkeypatch.setattr(SandboxManager, "root_dir", None)

    with pytest.raises(RuntimeError, match="not configured"):
        SandboxManager.get_root()
    with pytest.raises(RuntimeError, match="not configured"):
        SandboxManager.validate_path("file.txt")


def test_set_root_rejects_missing_or_non_directory_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(SandboxManager, "root_dir", None)
    missing = tmp_path / "missing"
    regular_file = tmp_path / "file.txt"
    regular_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        SandboxManager.set_root(str(missing))
    assert not missing.exists()

    with pytest.raises(NotADirectoryError, match="not a directory"):
        SandboxManager.set_root(str(regular_file))

    with pytest.raises(ValueError, match="non-empty"):
        SandboxManager.set_root("  ")


def test_set_root_accepts_an_existing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(SandboxManager, "root_dir", None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    configured = SandboxManager.set_root(str(workspace))

    assert configured == workspace.resolve()
    assert SandboxManager.get_root() == workspace.resolve()
    assert (
        SandboxManager.validate_path("file.txt") == (workspace / "file.txt").resolve()
    )
