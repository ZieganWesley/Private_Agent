from agent.permissions import (
    evaluate_permission_rules,
    permission_rule_matches,
)


def test_permission_rule_matches_tool_name_only():
    assert permission_rule_matches("edit_local_file", "edit_local_file", {}) is True
    assert permission_rule_matches("run_shell_command", "edit_local_file", {}) is False


def test_permission_rule_matches_arg_pattern():
    assert permission_rule_matches(
        "run_shell_command(npm test)", "run_shell_command", {"command": "npm test"}
    )
    assert not permission_rule_matches(
        "run_shell_command(npm test)", "run_shell_command", {"command": "rm -rf /"}
    )


def test_permission_rule_wildcard_arg_pattern():
    assert permission_rule_matches(
        "edit_local_file(src/*.py)", "edit_local_file", {"file_path": "src/main.py"}
    )
    assert not permission_rule_matches(
        "edit_local_file(src/*.py)", "edit_local_file", {"file_path": "docs/readme.md"}
    )


def test_permission_rule_empty_or_missing_pattern_matches_all():
    assert permission_rule_matches(
        "read_local_file()", "read_local_file", {"path": "x"}
    )
    assert permission_rule_matches(
        "read_local_file(  )", "read_local_file", {"path": "x"}
    )


def test_permission_rule_malformed_returns_false():
    assert not permission_rule_matches("not a rule", "read_local_file", {})
    assert not permission_rule_matches("", "read_local_file", {})


def test_evaluate_permission_rules_deny_precedence():
    decision, rule = evaluate_permission_rules(
        allow_rules=["run_shell_command(*)"],
        deny_rules=["run_shell_command(rm *)"],
        name="run_shell_command",
        args={"command": "rm -rf /tmp"},
    )
    assert decision == "deny"
    assert rule == "run_shell_command(rm *)"


def test_evaluate_permission_rules_allow_when_no_deny():
    decision, rule = evaluate_permission_rules(
        allow_rules=["edit_local_file(src/*.py)"],
        deny_rules=[],
        name="edit_local_file",
        args={"file_path": "src/main.py"},
    )
    assert decision == "allow"
    assert rule == "edit_local_file(src/*.py)"


def test_evaluate_permission_rules_none_when_no_match():
    decision, rule = evaluate_permission_rules(
        allow_rules=["edit_local_file(src/*.py)"],
        deny_rules=["run_shell_command(rm *)"],
        name="read_local_file",
        args={"path": "README.md"},
    )
    assert decision is None
    assert rule is None


def test_evaluate_permission_rules_handles_none_inputs():
    decision, rule = evaluate_permission_rules(None, None, "read_local_file", {})
    assert decision is None
    assert rule is None


def _console(answer: str = "yes", printed: list | None = None):
    from types import SimpleNamespace

    printed = printed if printed is not None else []
    return SimpleNamespace(
        input=lambda *a, **k: answer,
        print=lambda message="", *a, **k: printed.append(str(message)),
    )


def test_full_mode_requires_a_typed_confirmation():
    from agent.permissions import enforce_full_mode_confirmation

    assert (
        enforce_full_mode_confirmation("full", _console("yes"), lambda: True) == "full"
    )
    # A single keypress is not the deliberate confirmation full mode needs.
    assert enforce_full_mode_confirmation("full", _console("y"), lambda: True) == "auto"
    assert enforce_full_mode_confirmation("full", _console(""), lambda: True) == "auto"


def test_full_mode_confirmation_is_refused_without_a_terminal():
    from agent.permissions import enforce_full_mode_confirmation

    printed: list = []
    console = _console("yes", printed)

    assert enforce_full_mode_confirmation("full", console, lambda: False) == "auto"
    assert any("interactive confirmation" in message for message in printed)


def test_full_mode_confirmation_warning_names_what_it_grants():
    from agent.permissions import enforce_full_mode_confirmation

    printed: list = []
    enforce_full_mode_confirmation("full", _console("no", printed), lambda: True)

    warning = " ".join(printed)
    assert "shell commands" in warning
    assert "without another prompt" in warning


def test_non_full_modes_are_returned_without_asking():
    from agent.permissions import enforce_full_mode_confirmation

    prompts: list = []

    def fail_if_asked(*_args, **_kwargs):
        prompts.append("asked")
        raise AssertionError("non-full modes must not prompt")

    from types import SimpleNamespace

    console = SimpleNamespace(input=fail_if_asked, print=lambda *a, **k: None)
    for mode in ("auto", "manual", "monitored"):
        assert enforce_full_mode_confirmation(mode, console, lambda: True) == mode
    assert prompts == []
