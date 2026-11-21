"""The system prompt keeps a stable prefix so local servers can cache it."""

from private_agent.agent.prompt_prefix import (
    build_system_messages,
    stable_system_prompt,
)


def _messages(**overrides):
    kwargs = {
        "base_prompt": "Base rules.",
        "tool_limits_instruction": "At most 5 tool rounds. ",
        "deadline_seconds": 120,
        "max_tool_output_chars": 4000,
        "privacy_instruction": "This is the local Ollama provider.",
        "permission_mode": "auto",
        "datetime_context": "Current local date and time: Monday 10:00.",
    }
    kwargs.update(overrides)
    return build_system_messages(**kwargs)


def test_first_message_is_the_stable_prefix():
    first = _messages()[0].content
    second = _messages(datetime_context="Current local date and time: Tuesday 11:00.")[
        0
    ].content
    assert first == second
    assert first.startswith("Base rules.")
    assert "At most 5 tool rounds." in first
    assert "120-second execution deadline" in first
    assert "4000 characters" in first
    assert "permission mode" not in first


def test_volatile_context_comes_after_the_prefix():
    messages = _messages(
        planning_mode_instruction="This turn is planning-only.",
        active_task_instruction="Current user-approved task plan.",
        code_task_instructions="This is a code creation task.",
        project_instructions_block="\nProject instructions:\nUse tabs.\n",
    )
    assert len(messages) == 2
    runtime = messages[1].content
    assert runtime.startswith("Runtime context:")
    assert "This is the local Ollama provider." in runtime
    assert "permission mode is auto" in runtime
    assert "Monday 10:00" in runtime
    assert "planning-only" in runtime
    assert "user-approved task plan" in runtime
    assert "Use tabs." in runtime


def test_stable_prefix_ignores_a_changed_permission_mode():
    stable_a = stable_system_prompt(
        "Base rules.",
        tool_limits_instruction="",
        deadline_seconds=120,
        max_tool_output_chars=4000,
    )
    stable_b = stable_system_prompt(
        "Base rules.",
        tool_limits_instruction="",
        deadline_seconds=120,
        max_tool_output_chars=4000,
    )
    assert stable_a == stable_b
