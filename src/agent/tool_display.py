"""Rendering and redaction of tool calls and results for the CLI."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import ToolMessage
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .console import console, resolve_style

TOOL_RESULT_PREVIEW_CHARS = 500

# Longest argument value shown in a tool-request grid row before eliding.
TOOL_ARGUMENT_PREVIEW_CHARS = 200

_RECENT_TOOL_RESULTS: list[tuple[str, str]] = []
_MAX_RECENT_TOOL_RESULTS = 20


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


def _argument_value(value: Any) -> Text:
    """Render one argument value, eliding only genuinely huge payloads."""
    rendered = repr(value)
    if len(rendered) > TOOL_ARGUMENT_PREVIEW_CHARS:
        omitted = len(rendered) - TOOL_ARGUMENT_PREVIEW_CHARS
        rendered = (
            rendered[:TOOL_ARGUMENT_PREVIEW_CHARS] + f"\u2026 [{omitted} more chars]"
        )
    return Text(rendered, overflow="fold")


def _tool_request_renderable(tool_call: dict) -> Panel:
    """Build the tool-request panel: one grid row per argument."""
    name = str(tool_call.get("name") or "unknown")
    arguments = _redact_tool_arguments(tool_call.get("args", {}))
    table = Table.grid(padding=(0, 1))
    table.add_column(style=resolve_style("key"), justify="right", no_wrap=True)
    table.add_column(style=resolve_style("value"), overflow="fold")
    if isinstance(arguments, dict) and arguments:
        for key, value in arguments.items():
            table.add_row(str(key), _argument_value(value))
    else:
        table.add_row("args", _argument_value(arguments))
    return Panel(
        table,
        title=Text(f"[Tool Request] {name}", style=resolve_style("accent")),
        border_style=resolve_style("border"),
        padding=(0, 1),
    )


def _display_tool_request(tool_call: dict) -> None:
    console.print(_tool_request_renderable(tool_call))


def _display_tool_error(tool_name: str, details: str) -> None:
    console.print(
        Panel(
            Text(details),
            title=Text(
                f"\u2717 Tool '{tool_name}' returned an error",
                style=resolve_style("error"),
            ),
            border_style=resolve_style("border.error"),
        )
    )


def _render_tool_result_text(result: str) -> Text:
    """Render a tool result, colouring diff-like output when detected.

    Unified diffs are rendered with ``rich.syntax.Syntax`` so added and
    removed lines receive the diff lexer's colours.
    """
    lines = result.splitlines()
    looks_like_diff = any(
        line.startswith(("+++", "---", "@@", "+", "-")) for line in lines
    ) and any(line.startswith(("+++", "---")) for line in lines)
    if looks_like_diff:
        return Syntax(result, "diff", theme="ansi_dark", word_wrap=False)
    text = Text()
    for index, line in enumerate(lines):
        if index:
            text.append("\n")
        text.append(line)
    return text


def _display_tool_result(tool_name: str, result: str) -> None:
    preview = result[:TOOL_RESULT_PREVIEW_CHARS]
    truncated = len(result) > TOOL_RESULT_PREVIEW_CHARS
    if truncated:
        preview += "\n[Preview truncated; full result was sent to the model.]"
    if not preview.strip():
        preview = "(Tool completed with no textual output.)"
    symbol = "\u2713" if not _tool_result_is_error(result) else "\u2717"
    style = resolve_style("success" if symbol == "\u2713" else "error")
    console.print(
        Panel(
            _render_tool_result_text(preview),
            title=Text(f"{symbol} Tool output: {tool_name}", style=style),
            border_style=resolve_style("border"),
        )
    )
    record_tool_result(tool_name, result)


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


def record_tool_result(tool_name: str, result: str) -> None:
    """Remember a full tool result so it can be expanded later (G11)."""
    _RECENT_TOOL_RESULTS.append((tool_name, result))
    if len(_RECENT_TOOL_RESULTS) > _MAX_RECENT_TOOL_RESULTS:
        del _RECENT_TOOL_RESULTS[: len(_RECENT_TOOL_RESULTS) - _MAX_RECENT_TOOL_RESULTS]


def expand_last_tool_result(index: int = -1) -> str | None:
    """Return the full text of a stored tool result, or None if unavailable."""
    if not _RECENT_TOOL_RESULTS:
        return None
    if index < 0:
        index = len(_RECENT_TOOL_RESULTS) + index
    if index < 0 or index >= len(_RECENT_TOOL_RESULTS):
        return None
    _name, result = _RECENT_TOOL_RESULTS[index]
    return result


def list_recent_tool_results() -> list[tuple[str, str]]:
    """Return (name, full_result) pairs for recently displayed tool results."""
    return list(_RECENT_TOOL_RESULTS)


class _LazyProxy:
    __slots__ = ("_module", "_attr", "_value", "_loaded")

    def __init__(self, module_path, attr_name):
        self._module = module_path
        self._attr = attr_name
        self._value = None
        self._loaded = False

    def _load(self):
        if not self._loaded:
            import importlib

            module = importlib.import_module(self._module)
            self._value = getattr(module, self._attr)
            self._loaded = True
        return self._value

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._load(), name, value)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __bool__(self) -> bool:
        return bool(self._load())

    def __repr__(self):
        return repr(self._load())


captured_image_message = _LazyProxy("tools.media", "captured_image_message")
