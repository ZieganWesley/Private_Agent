"""Message-shape normalization for strict chat templates.

Qwen3.5+ templates served by llama.cpp (and Ollama) raise
``System message must be at the beginning`` for any system-role message that is
not part of the leading system block. OpenAI-compatible clients may legally send
several system messages, so the request boundary adapts the list before it is
handed to the model instead of relying on every caller getting the shape right.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

LATE_SYSTEM_PREFIX = "[System note]"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return str(content or "")


def _is_system(message: Any) -> bool:
    return (
        isinstance(message, SystemMessage) or getattr(message, "type", None) == "system"
    )


def normalize_system_messages(messages: Iterable[Any]) -> list[Any]:
    """Return a message list that a strict chat template can render.

    Leading system messages are merged into one, preserving their order. A
    system message that appears after the conversation has started is kept in
    place as a labelled user message, which every OpenAI-compatible server
    accepts and which keeps the instruction adjacent to the turn it applies to.
    """
    result: list[Any] = []
    leading_system_text: list[str] = []
    leading = True
    for message in messages:
        if _is_system(message):
            text = _message_text(message)
            if leading:
                if text:
                    leading_system_text.append(text)
                continue
            label = LATE_SYSTEM_PREFIX
            result.append(HumanMessage(content=f"{label}\n{text}" if text else label))
            continue
        leading = False
        result.append(message)
    if leading_system_text:
        result.insert(0, SystemMessage(content="\n\n".join(leading_system_text)))
    return result


__all__ = ["LATE_SYSTEM_PREFIX", "normalize_system_messages"]
