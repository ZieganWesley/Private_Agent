"""Responses API schema conversion and stream parsing (P5b).

Pure helpers keep the transport small and let tests drive the exact shapes
without a live endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TOOL_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


def tool_definitions(tools: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Convert LangChain tools or OpenAI-style dicts to Responses tool defs."""
    definitions: list[dict[str, Any]] = []
    for tool in tools or []:
        name = description = None
        parameters: Any = None
        if isinstance(tool, dict):
            source = (
                tool.get("function") if isinstance(tool.get("function"), dict) else tool
            )
            name = source.get("name")
            description = source.get("description")
            parameters = (
                source.get("parameters")
                or source.get("input_schema")
                or source.get("schema")
            )
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            schema = getattr(tool, "args_schema", None)
            if schema is not None:
                dump = getattr(schema, "model_json_schema", None)
                if callable(dump):
                    parameters = dump()
                elif isinstance(schema, dict):
                    parameters = schema
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(parameters, dict) or not parameters:
            parameters = dict(DEFAULT_TOOL_PARAMETERS)
        definitions.append(
            {
                "type": "function",
                "name": name,
                "description": str(description or ""),
                "parameters": parameters,
            }
        )
    return definitions


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return str(content or "")


def input_items(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Map internal chat messages to Responses ``input`` items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.__class__.__name__.replace("Message", "").lower()
        if role == "ai":
            role = "assistant"
        if role == "human":
            role = "user"
        if role == "system":
            role = "system"
        text = _content_text(getattr(message, "content", ""))
        if text:
            items.append(
                {"role": role, "content": [{"type": "input_text", "text": text}]}
            )
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id") or call.get("call_id")
            arguments = call.get("args", {})
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": call.get("name"),
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments,
                }
            )
    return items


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_call(item: dict[str, Any]) -> dict[str, Any] | None:
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    call_id = item.get("call_id") or item.get("id")
    return {
        "name": name,
        "args": _parse_arguments(item.get("arguments")),
        "id": str(call_id) if call_id is not None else None,
    }


def reply_from_body(body: Any) -> dict[str, Any] | None:
    """Return ``{text, tool_calls, usage, model}`` or None for invalid shapes."""
    if not isinstance(body, dict):
        return None
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                call = _tool_call(item)
                if call is not None:
                    tool_calls.append(call)
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
    output_text = body.get("output_text")
    if not text_parts and isinstance(output_text, str):
        text_parts.append(output_text)
    if not text_parts and not tool_calls:
        return None
    usage = body.get("usage")
    model = body.get("model")
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls or None,
        "usage": usage if isinstance(usage, dict) else None,
        "model": model if isinstance(model, str) else None,
    }


def iter_sse_events(lines: Iterable[Any]) -> Iterable[dict[str, Any]]:
    """Yield decoded JSON payloads from an SSE line stream."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        text = str(line or "").strip()
        if not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if isinstance(payload, dict):
            yield payload


@dataclass
class ResponsesStreamState:
    """Accumulate streamed Responses events into text and tool calls."""

    text_parts: list[str] = field(default_factory=list)
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    completed: dict[str, Any] | None = None

    def _call_for(self, item_id: Any) -> dict[str, Any] | None:
        key = str(item_id)
        call = self.calls.get(key)
        if call is None:
            call = {"id": key, "name": "", "arguments": ""}
            self.calls[key] = call
            self.order.append(key)
        return call

    def handle(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply one event and return normalized deltas for the caller."""
        kind = event.get("type")
        if kind == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self.text_parts.append(delta)
                return [{"type": "text_delta", "text": delta}]
            return []
        if kind == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                key = str(item.get("id") or item.get("call_id") or len(self.order))
                call = self.calls.setdefault(
                    key, {"id": item.get("call_id") or key, "name": "", "arguments": ""}
                )
                if key not in self.order:
                    self.order.append(key)
                if isinstance(item.get("name"), str):
                    call["name"] = item["name"]
                if isinstance(item.get("arguments"), str):
                    call["arguments"] = item["arguments"]
                return [
                    {
                        "type": "tool_call_start",
                        "index": self.order.index(key),
                        "id": call.get("id"),
                        "name": call.get("name", ""),
                    }
                ]
            return []
        if kind == "response.function_call_arguments.delta":
            call = self._call_for(event.get("item_id"))
            delta = event.get("delta")
            if isinstance(delta, str) and delta and call is not None:
                call["arguments"] += delta
                return [
                    {
                        "type": "tool_call_delta",
                        "index": self.order.index(str(event.get("item_id"))),
                        "id": call.get("id"),
                        "arguments": delta,
                    }
                ]
            return []
        if kind == "response.function_call_arguments.done":
            call = self._call_for(event.get("item_id"))
            arguments = event.get("arguments")
            if call is not None and isinstance(arguments, str):
                call["arguments"] = arguments
            return []
        if kind == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                key = str(item.get("id") or item.get("call_id") or "")
                call = self.calls.get(key) or self._call_for(key)
                if call is not None:
                    if isinstance(item.get("name"), str):
                        call["name"] = item["name"]
                    if isinstance(item.get("arguments"), str):
                        call["arguments"] = item["arguments"]
            return []
        if kind == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                self.completed = response
            return [{"type": "completed", "body": self.completed}]
        if kind in {"response.failed", "error"}:
            message = event.get("message") or event.get("error") or kind
            return [{"type": "error", "error": str(message)}]
        return []

    def result(self) -> dict[str, Any] | None:
        parsed = reply_from_body(self.completed) if self.completed else None
        if parsed is not None:
            return parsed
        tool_calls = []
        for key in self.order:
            call = self.calls.get(key) or {}
            if not call.get("name"):
                continue
            tool_calls.append(
                {
                    "name": call["name"],
                    "args": _parse_arguments(call.get("arguments")),
                    "id": call.get("id"),
                }
            )
        text = "".join(self.text_parts)
        if not text and not tool_calls:
            return None
        return {
            "text": text,
            "tool_calls": tool_calls or None,
            "usage": None,
            "model": None,
        }


__all__ = [
    "ResponsesStreamState",
    "input_items",
    "iter_sse_events",
    "reply_from_body",
    "tool_definitions",
]
