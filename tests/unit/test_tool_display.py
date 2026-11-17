"""Behavioural tests for the shared tool-call rendering helpers."""

import io

from langchain_core.messages import ToolMessage
from rich.console import Console

from private_agent.agent import tool_display


def test_argument_redaction_masks_nested_credentials():
    arguments = {
        "query": "public profile",
        "api_key": "must-not-be-visible",
        "nested": {"password": "hunter2", "options": [{"access_token": "abc123"}]},
        "count": 3,
    }

    redacted = tool_display._redact_tool_arguments(arguments)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["options"][0]["access_token"] == "[REDACTED]"
    assert redacted["query"] == "public profile"
    assert redacted["count"] == 3


def test_sensitive_values_are_collected_only_under_sensitive_keys():
    arguments = {
        "name": "report",
        "secret": "s3cr3t-value",
        "headers": {"Authorization": "Bearer token-value"},
        "items": ["plain-value"],
    }

    values = tool_display._sensitive_argument_values(arguments)

    assert "s3cr3t-value" in values
    assert "Bearer token-value" in values
    assert "plain-value" not in values


def test_redaction_replaces_every_sensitive_occurrence_longest_first():
    arguments = {"token": "abcdef", "passwords": ["abcdefgh"]}

    text = "value abcdefgh then abcdef"

    redacted = tool_display._redact_sensitive_values(text, arguments)

    assert redacted == "value [REDACTED] then [REDACTED]"


def test_log_redactions_include_sensitive_values_and_short_strings_are_unique():
    arguments = {"token": "session-token", "note": "visible-note", "tiny": "abc"}

    redactions = tool_display._tool_log_redactions(arguments)

    assert "session-token" in redactions
    assert "visible-note" in redactions
    assert "abc" not in redactions
    assert len(redactions) == len(set(redactions))


def test_tool_request_panel_uses_shared_console(monkeypatch):
    output = io.StringIO()
    captured = Console(file=output, width=100, force_terminal=False)
    printed = []
    monkeypatch.setattr(
        tool_display.console,
        "print",
        lambda *values, **_kwargs: printed.append(values),
    )

    tool_display._display_tool_request(
        {"name": "web_search", "args": {"query": "public", "api_key": "hidden"}}
    )

    assert printed[0][0] == "[cyan][Tool Request][/cyan]"
    summary = printed[0][1].plain
    assert summary.startswith("web_search")
    assert "[REDACTED]" in summary
    assert "hidden" not in summary
    assert captured.file is output


def test_tool_result_preview_is_bounded_and_labeled(monkeypatch):
    printed = []
    monkeypatch.setattr(
        tool_display.console,
        "print",
        lambda *values, **_kwargs: printed.append(values),
    )

    tool_display._display_tool_result(
        "search_workspace",
        "x" * (tool_display.TOOL_RESULT_PREVIEW_CHARS + 20),
    )

    panel = printed[0][0]
    assert panel.title.plain == "Tool output: search_workspace"
    body = panel.renderable.plain
    assert body.startswith("x" * tool_display.TOOL_RESULT_PREVIEW_CHARS)
    assert "Preview truncated" in body


def test_empty_tool_result_reports_no_textual_output(monkeypatch):
    printed = []
    monkeypatch.setattr(
        tool_display.console,
        "print",
        lambda *values, **_kwargs: printed.append(values),
    )

    tool_display._display_tool_result("noop", "")

    assert printed[0][0].renderable.plain == "(Tool completed with no textual output.)"


def test_tool_result_error_detection_matches_failures_only():
    assert tool_display._tool_result_is_error("Error: permission denied")
    assert tool_display._tool_result_is_error("Web search failed for the query")
    assert tool_display._tool_result_is_error("Exit code: 2")
    assert not tool_display._tool_result_is_error("Saved 3 files successfully")
    assert not tool_display._tool_result_is_error("errors were handled gracefully")


def test_local_image_input_error_reports_missing_image_reference():
    tool_calls = [{"name": "capture_webcam_image", "id": "call-1"}]
    tool_messages = [ToolMessage(content="No camera available", tool_call_id="call-1")]

    assert (
        tool_display._local_image_input_error(tool_calls, tool_messages)
        == "No camera available"
    )


def test_local_image_input_error_accepts_valid_reference_and_declined_capture():
    tool_calls = [
        {"name": "capture_webcam_image", "id": "call-1"},
        {"name": "load_workspace_image", "id": "call-2"},
    ]
    tool_messages = [
        ToolMessage(content="camera-image:" + "a" * 32, tool_call_id="call-1"),
        ToolMessage(content="Media capture was declined", tool_call_id="call-2"),
    ]

    assert tool_display._local_image_input_error(tool_calls, tool_messages) is None


def test_captured_images_from_tool_results_are_appended(monkeypatch):
    appended = []

    def fake_captured_image_message(reference):
        appended.append(reference)
        return f"image:{reference}"

    monkeypatch.setattr(
        tool_display, "captured_image_message", fake_captured_image_message
    )
    messages = [
        ToolMessage(content="workspace-image:" + "b" * 32, tool_call_id="call-1")
    ]
    collected = []

    tool_display._append_captured_images_from_tool_results(collected, messages)

    assert collected == ["image:workspace-image:" + "b" * 32]
    assert appended == ["workspace-image:" + "b" * 32]
