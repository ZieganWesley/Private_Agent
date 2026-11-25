"""LangChain-shaped model that speaks /v1/completions or /v1/responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from .transports import CompletionsClient, ResponsesClient


def _text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return str(content or "")


def _flatten(messages: Iterable[BaseMessage]) -> tuple[str | None, str]:
    instructions: list[str] = []
    turns: list[str] = []
    for message in messages:
        text = _text(message)
        if message.__class__.__name__ == "SystemMessage":
            instructions.append(text)
        else:
            turns.append(f"{message.__class__.__name__}: {text}")
    return ("\n\n".join(instructions) or None, "\n\n".join(turns))


class TransportChatModel:
    """Call an optional OpenAI-compatible route through a chat-model interface."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_mode: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        client_factory: Any = None,
    ):
        if api_mode not in {"completions", "responses"}:
            raise ValueError(f"Unsupported transport mode: {api_mode}")
        self.base_url = base_url
        self.model_name = model_name
        self.api_mode = api_mode
        self.api_key = api_key
        self.timeout = timeout
        self.client_factory = client_factory

    def bind_tools(self, tools: Any):
        """These routes have no tool-call contract; keep the model usable."""
        return self

    def _call(self, messages: Iterable[BaseMessage]) -> str:
        instructions, prompt = _flatten(messages)
        if self.api_mode == "completions":
            if instructions:
                prompt = f"{instructions}\n\n{prompt}" if prompt else instructions
            client = CompletionsClient(
                self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                client_factory=self.client_factory,
            )
            result = client.complete(prompt, model=self.model_name)
        else:
            client = ResponsesClient(
                self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                client_factory=self.client_factory,
            )
            result = client.respond(
                prompt, model=self.model_name, instructions=instructions
            )
        if result is None:
            raise RuntimeError(
                f"The {self.api_mode} transport returned no usable response."
            )
        return result.text

    async def ainvoke(self, messages: Iterable[BaseMessage]) -> AIMessage:
        return AIMessage(content=await asyncio.to_thread(self._call, messages))

    async def astream(self, messages: Iterable[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        yield AIMessageChunk(content=await asyncio.to_thread(self._call, messages))


__all__ = ["TransportChatModel"]
