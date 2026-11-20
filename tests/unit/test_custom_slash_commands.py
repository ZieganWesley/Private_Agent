from private_agent.agent import interactive


def _patch_home(monkeypatch, home_root):
    monkeypatch.setattr(
        interactive.pathlib.Path,
        "home",
        classmethod(lambda cls: home_root),
    )


def test_load_custom_slash_commands_empty(tmp_path, monkeypatch):
    _patch_home(monkeypatch, tmp_path / "home")
    commands = interactive.load_custom_slash_commands(tmp_path / "workspace")
    assert commands == {}


def test_load_custom_slash_commands_project_and_user(tmp_path, monkeypatch):
    home_root = tmp_path / "home"
    user_dir = home_root / ".private_agent" / "commands"
    user_dir.mkdir(parents=True)
    (user_dir / "hello.md").write_text("Hello from user", encoding="utf-8")

    workspace = tmp_path / "workspace"
    project_dir = workspace / ".private_agent" / "commands"
    project_dir.mkdir(parents=True)
    (project_dir / "review.md").write_text("Review code", encoding="utf-8")
    (project_dir / "hello.md").write_text("Hello from project", encoding="utf-8")

    _patch_home(monkeypatch, home_root)
    commands = interactive.load_custom_slash_commands(workspace)

    assert commands["/hello"] == "Hello from project"
    assert commands["/review"] == "Review code"


def test_load_custom_slash_commands_ignores_non_markdown(tmp_path, monkeypatch):
    home_root = tmp_path / "home"
    user_dir = home_root / ".private_agent" / "commands"
    user_dir.mkdir(parents=True)
    (user_dir / "note.txt").write_text("ignored", encoding="utf-8")
    (user_dir / "cmd.md").write_text("run", encoding="utf-8")

    _patch_home(monkeypatch, home_root)
    commands = interactive.load_custom_slash_commands(tmp_path / "workspace")
    assert list(commands) == ["/cmd"]


def test_load_custom_slash_commands_handles_missing_dir(tmp_path, monkeypatch):
    _patch_home(monkeypatch, tmp_path / "nohome")
    commands = interactive.load_custom_slash_commands(tmp_path / "noworkspace")
    assert commands == {}


def test_load_custom_slash_commands_skips_unreadable(tmp_path, monkeypatch):
    home_root = tmp_path / "home"
    user_dir = home_root / ".private_agent" / "commands"
    user_dir.mkdir(parents=True)
    (user_dir / "bad.md").write_bytes(b"\xff\xfe")
    _patch_home(monkeypatch, home_root)
    # Should not raise; may skip or decode with errors ignored
    commands = interactive.load_custom_slash_commands(tmp_path / "workspace")
    assert isinstance(commands, dict)
