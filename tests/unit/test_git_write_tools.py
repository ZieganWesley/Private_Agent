import subprocess
from pathlib import Path

from private_agent.sandbox import SandboxManager
from private_agent.tools import filesystem


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)


def test_create_git_branch(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.setattr(SandboxManager, "root_dir", None)
    SandboxManager.set_root(str(tmp_path))
    result = filesystem.create_git_branch.invoke({"branch_name": "feature/test"})
    assert "Created and checked out branch" in result
    assert "feature/test" in subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=tmp_path, text=True
    )


def test_stage_and_commit_git_changes(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    SandboxManager.set_root(str(tmp_path))
    (tmp_path / "new.txt").write_text("new\n", encoding="utf-8")
    assert "Staged" in filesystem.stage_git_changes.invoke({"paths": ["new.txt"]})
    assert "initial" not in filesystem.commit_git_changes.invoke(
        {"message": "add new file"}
    )
    log = subprocess.check_output(["git", "log", "--oneline"], cwd=tmp_path, text=True)
    assert "add new file" in log


def test_create_git_branch_requires_repository(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
    result = filesystem.create_git_branch.invoke({"branch_name": "feature/test"})
    assert "No Git repository" in result


def test_create_pull_request_without_gh(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    SandboxManager.set_root(str(tmp_path))

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(filesystem.subprocess, "run", _raise)
    result = filesystem.create_pull_request.invoke({"title": "title", "body": "body"})
    assert "GitHub CLI" in result or "Pull request creation failed" in result
