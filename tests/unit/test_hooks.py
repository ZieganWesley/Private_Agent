from private_agent import hooks


def test_load_hooks_empty():
    assert hooks.load_hooks({}) == {}
    assert hooks.load_hooks(None) == {}


def test_load_hooks_normalizes_strings_and_dicts():
    config = {
        "hooks": {
            "PreToolUse": ["echo pre", {"command": "echo second", "matcher": "edit_*"}],
            "PostToolUse": {"command": "echo post", "matcher": "*"},
        }
    }
    loaded = hooks.load_hooks(config)
    assert len(loaded["PreToolUse"]) == 2
    assert loaded["PreToolUse"][0]["command"] == "echo pre"
    assert loaded["PreToolUse"][1]["matcher"] == "edit_*"
    assert loaded["PostToolUse"][0]["command"] == "echo post"


def test_load_hooks_claude_style_nested():
    config = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Edit",
                    "hooks": [{"type": "command", "command": "echo nested"}],
                }
            ]
        }
    }
    loaded = hooks.load_hooks(config)
    assert loaded["PreToolUse"][0]["command"] == "echo nested"
    assert loaded["PreToolUse"][0]["matcher"] == "Edit"


def test_load_hooks_from_agent_section():
    config = {"agent": {"hooks": {"SessionStart": ["echo start"]}}}
    loaded = hooks.load_hooks(config)
    assert "SessionStart" in loaded


def test_run_hooks_allows_zero_exit():
    result = hooks.run_hooks(
        "PreToolUse",
        hooks={"PreToolUse": [{"command": "echo ok"}]},
        tool_name="edit_local_file",
        tool_args={"file_path": "a.py"},
    )
    assert len(result) == 1
    assert result[0].returncode == 0
    assert result[0].blocked is False


def test_run_hooks_blocks_nonzero_exit():
    result = hooks.run_hooks(
        "PreToolUse",
        hooks={"PreToolUse": [{"command": "exit 1"}]},
        tool_name="edit_local_file",
        tool_args={},
    )
    assert result[0].blocked is True
    assert hooks.hooks_blocked(result) is not None


def test_run_hooks_blocks_json_decision():
    cmd = 'python3 -c \'import json,sys; print(json.dumps({"decision":"block","reason":"nope"}))\''
    result = hooks.run_hooks(
        "PreToolUse",
        hooks={"PreToolUse": [{"command": cmd}]},
        tool_name="edit_local_file",
        tool_args={},
    )
    assert result[0].blocked is True
    assert "nope" in (hooks.hooks_blocked(result) or "")


def test_run_hooks_matcher_filters():
    result = hooks.run_hooks(
        "PreToolUse",
        hooks={"PreToolUse": [{"command": "exit 1", "matcher": "run_shell_command"}]},
        tool_name="edit_local_file",
        tool_args={},
    )
    assert result == []


def test_run_hooks_timeout_blocks_pre_tool():
    result = hooks.run_hooks(
        "PreToolUse",
        hooks={"PreToolUse": [{"command": "sleep 5", "timeout": 1}]},
        tool_name="edit_local_file",
        tool_args={},
        timeout=10,
    )
    assert result[0].timed_out is True
    assert result[0].blocked is True


def test_run_hooks_post_tool_does_not_block():
    result = hooks.run_hooks(
        "PostToolUse",
        hooks={"PostToolUse": [{"command": "exit 1"}]},
        tool_name="edit_local_file",
        tool_args={},
    )
    assert result[0].blocked is False


def test_run_hooks_handles_missing_command():
    result = hooks.run_hooks(
        "PreToolUse",
        hooks={"PreToolUse": [{"command": "definitely-not-a-command-xyz"}]},
        tool_name="edit_local_file",
        tool_args={},
    )
    assert len(result) == 1
    # Nonzero exit or failure to start should be recorded
    assert result[0].returncode not in (0, None) or result[0].reason


def test_run_session_start_hooks_alias():
    result = hooks.run_session_start_hooks(
        hooks={"SessionStart": [{"command": "echo started"}]},
    )
    assert len(result) == 1
    assert result[0].returncode == 0


def test_hooks_blocked_none():
    assert hooks.hooks_blocked([]) is None
    assert hooks.hooks_blocked(None) is None
