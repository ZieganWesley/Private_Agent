import pytest

from sandbox import SandboxManager


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


def test_sandbox_network_permission_defaults_disabled(monkeypatch):
    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    assert SandboxManager.is_network_allowed() is False
    assert SandboxManager.allow_network is False


def test_set_network_allowed_updates_both_aliases(monkeypatch):
    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    assert SandboxManager.set_network_allowed(True) is True
    assert SandboxManager.allow_network is True
    assert SandboxManager.network_allowed is True
    assert SandboxManager.is_network_allowed() is True

    assert SandboxManager.set_network_permission(False) is False
    assert SandboxManager.allow_network is False
    assert SandboxManager.network_allowed is False
    assert SandboxManager.is_network_allowed() is False


def test_set_network_allowed_accepts_common_safe_values(monkeypatch):
    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    for value in (1, "true", "yes", "on", "enabled"):
        assert SandboxManager.set_network_allowed(value) is True
        assert SandboxManager.is_network_allowed() is True

    for value in (0, "false", "no", "off", "disabled"):
        assert SandboxManager.set_network_allowed(value) is False
        assert SandboxManager.is_network_allowed() is False


def test_set_network_allowed_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    for value in ("maybe", 2, object(), None, [], {}):
        with pytest.raises(ValueError):
            SandboxManager.set_network_allowed(value)
    assert SandboxManager.is_network_allowed() is False


def test_enable_and_disable_network_helpers(monkeypatch):
    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    SandboxManager.enable_network()
    assert SandboxManager.is_network_allowed() is True

    SandboxManager.disable_network()
    assert SandboxManager.is_network_allowed() is False


def test_sandbox_network_permission_fails_closed_on_attribute_errors(monkeypatch):
    class BrokenSandbox:
        @classmethod
        def is_network_allowed(cls):
            raise RuntimeError("boom")

    monkeypatch.setattr("tools.shell.SandboxManager", BrokenSandbox)
    from tools.shell import _sandbox_network_allowed

    assert _sandbox_network_allowed() is False


def test_direct_invalid_network_attribute_fails_closed(monkeypatch):
    monkeypatch.setattr(SandboxManager, "allow_network", "not-a-bool")
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    assert SandboxManager.is_network_allowed() is False

    # One tampered alias must not be overridden by the other alias being true.
    monkeypatch.setattr(SandboxManager, "network_allowed", True)
    assert SandboxManager.is_network_allowed() is False


def test_sandbox_network_allowed_uses_global_permission_without_code_task(monkeypatch):
    from tools import get_tool_context
    from tools.shell import _sandbox_network_allowed

    context = get_tool_context()
    monkeypatch.setattr(context, "code_task", None)
    monkeypatch.setattr(SandboxManager, "allow_network", True)
    monkeypatch.setattr(SandboxManager, "network_allowed", True)

    assert _sandbox_network_allowed() is True


def test_sandbox_network_allowed_code_task_explicit_false_overrides_global(monkeypatch):
    from tools import get_tool_context
    from tools.shell import _sandbox_network_allowed

    class CodeTaskWithoutNetwork:
        allow_network = False

    context = get_tool_context()
    monkeypatch.setattr(context, "code_task", CodeTaskWithoutNetwork())
    monkeypatch.setattr(SandboxManager, "allow_network", True)
    monkeypatch.setattr(SandboxManager, "network_allowed", True)

    assert _sandbox_network_allowed() is False


def test_sandbox_network_config_key_and_permissions_alias():
    from config.settings import Settings

    assert (
        Settings.from_config({"sandbox": {"network": True}}).sandbox_network_default
        is True
    )
    assert (
        Settings.from_config(
            {"permissions": {"sandbox_network": True}}
        ).sandbox_network_default
        is True
    )
    assert (
        Settings.from_config({"sandbox": {"network": "not-a-bool"}}).sandbox_network_default
        is False
    )


def test_startup_network_permission_resolution(monkeypatch):
    from agent.runtime import _apply_sandbox_network_permission

    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)

    assert _apply_sandbox_network_permission(False, {}) is False
    assert _apply_sandbox_network_permission(True, {}) is True
    assert _apply_sandbox_network_permission(True, {"PRIVATE_AGENT_SANDBOX_NETWORK": "false"}) is False
    assert _apply_sandbox_network_permission(False, {"PRIVATE_AGENT_SANDBOX_NETWORK": "1"}) is True
    assert _apply_sandbox_network_permission(True, {"PRIVATE_AGENT_SANDBOX_NETWORK": ""}) is False
    assert _apply_sandbox_network_permission(False, {"PRIVATE_AGENT_SANDBOX_NETWORK": "garbage"}) is False
    assert _apply_sandbox_network_permission(object(), {}) is False
    assert SandboxManager.is_network_allowed() is False


def test_cli_sandbox_network_flag_reaches_startup_resolution(monkeypatch):
    import cli
    from agent import runtime
    from utils import run_logging

    received = []
    monkeypatch.setattr(SandboxManager, "allow_network", False)
    monkeypatch.setattr(SandboxManager, "network_allowed", False)
    monkeypatch.delenv("PRIVATE_AGENT_SANDBOX_NETWORK", raising=False)
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["private-agent", "--sandbox-network"],
    )
    monkeypatch.setattr(run_logging, "start_debug_logging", lambda: None)
    monkeypatch.setattr(run_logging, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_logging, "stop_debug_logging", lambda: None)
    monkeypatch.setattr(
        runtime,
        "run_agent_cli",
        lambda **kwargs: received.append(
            runtime._apply_sandbox_network_permission()
        ),
    )

    cli.main()

    assert received == [True]
    assert SandboxManager.is_network_allowed() is True
