"""Rendering and redaction of tool calls and results for the CLI."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import ToolMessage
from rich.panel import Panel
from rich.text import Text

from ..tools.media import captured_image_message
from .console import console

TOOL_RESULT_PREVIEW_CHARS = 500


def _redact_tool_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if re.search(
                r"(password|secret|token|api.?key|authorization|credential)",
                str(key),
                re.IGNORECASE,
            ):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_tool_arguments(item)
        return redacted
    if isinstance(value, list):
        return [_redact_tool_arguments(item) for item in value]
    return value


def _sensitive_argument_values(
    value: Any, *, sensitive_parent: bool = False
) -> list[str]:
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            sensitive = sensitive_parent or bool(
                re.search(
                    r"(password|secret|token|api.?key|authorization|credential)",
                    str(key),
                    re.IGNORECASE,
                )
            )
            values.extend(_sensitive_argument_values(item, sensitive_parent=sensitive))
        return values
    if isinstance(value, list):
        return [
            secret
            for item in value
            for secret in _sensitive_argument_values(
                item, sensitive_parent=sensitive_parent
            )
        ]
    if (
        sensitive_parent
        and isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    ):
        secret = str(value)
        return [secret] if secret else []
    return []


def _redact_sensitive_values(text: str, arguments: Any) -> str:
    secrets = sorted(
        set(_sensitive_argument_values(arguments)),
        key=len,
        reverse=True,
    )
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    return text


def _tool_log_redactions(arguments: Any) -> tuple[str, ...]:
    values = _sensitive_argument_values(arguments)

    def collect_strings(value: Any) -> None:
        if isinstance(value, dict):
            for nested_value in value.values():
                collect_strings(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                collect_strings(nested_value)
        elif isinstance(value, str) and len(value) >= 4:
            values.append(value)

    collect_strings(arguments)
    return tuple(dict.fromkeys(values))


def _display_tool_request(tool_call: dict) -> None:
    name = str(tool_call.get("name") or "unknown")
    arguments = _redact_tool_arguments(tool_call.get("args", {}))
    summary = repr(arguments)
    if len(summary) > 300:
        summary = summary[:297] + "..."
    console.print(
        "[cyan][Tool Request][/cyan]",
        Text(f"{name} {summary}", overflow="ellipsis"),
    )


def _display_tool_error(tool_name: str, details: str) -> None:
    console.print(
        Panel(
            Text(details),
            title=Text(f"Tool '{tool_name}' returned an error"),
            border_style="red",
        )
    )


def _display_tool_result(tool_name: str, result: str) -> None:
    preview = result[:TOOL_RESULT_PREVIEW_CHARS]
    if len(result) > TOOL_RESULT_PREVIEW_CHARS:
        preview += "\n[Preview truncated; full result was sent to the model.]"
    if not preview:
        preview = "(Tool completed with no textual output.)"
    console.print(
        Panel(
            Text(preview),
            title=Text(f"Tool output: {tool_name}"),
            border_style="blue",
        )
    )


def _tool_result_is_error(result: str) -> bool:
    return bool(
        re.search(
            r"(?im)^\s*(?:"
            r"error\b|failed\b|failure\b|could not\b|unable to\b|"
            r"(?:web search|fetch|download|execution|task|checkpoint|"
            r"final checkpoint|git [\w-]+|video processing|image processing|"
            r"camera capture|microphone recording|offline transcription)"
            r"\s+(?:failed|failure)\b"
            r")|(?:\bexit code:\s*[1-9]\b|\btraceback\b)",
            result,
        )
    )


def _local_image_input_error(
    tool_calls: list[dict],
    tool_messages: list[ToolMessage],
) -> str | None:
    """Return a local image tool result when no usable image was produced."""
    messages_by_call_id = {
        message.tool_call_id: str(message.content) for message in tool_messages
    }
    for call in tool_calls:
        if call.get("name") not in {
            "capture_webcam_image",
            "load_workspace_image",
            "load_workspace_video",
        }:
            continue
        result = messages_by_call_id.get(call.get("id"), "")
        if "Media capture was declined" in result:
            continue
        if not re.search(r"(?:camera|workspace|video)-image:[0-9a-f]{32}", result):
            return result or "The local image tool returned no result."
    return None


def _append_captured_images_from_tool_results(
    messages: list,
    tool_messages: list[ToolMessage],
) -> None:
    for tool_message in tool_messages:
        references = re.finditer(
            r"(camera-image|workspace-image|video-image):([0-9a-f]{32})",
            str(tool_message.content),
        )
        for reference in references:
            image_message = captured_image_message(
                f"{reference.group(1)}:{reference.group(2)}"
            )
            if image_message is not None:
                messages.append(image_message)
