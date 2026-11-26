"""LangChain-shaped model that speaks /v1/completions or /v1/responses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from .transports import CompletionsClient, ResponsesClient, TransportResult


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
        self.tools: Any = None

    def bind_tools(self, tools: Any):
        """Remember tool definitions for the Responses transport."""
        self.tools = tools
        return self

    def _client(self) -> CompletionsClient | ResponsesClient:
        if self.api_mode == "completions":
            return CompletionsClient(
                self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                client_factory=self.client_factory,
            )
        return ResponsesClient(
            self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            client_factory=self.client_factory,
        )

    def _invoke(self, messages: Iterable[BaseMessage]) -> TransportResult:
        instructions, prompt = _flatten(messages)
        if self.api_mode == "completions":
            if instructions:
                prompt = f"{instructions}\n\n{prompt}" if prompt else instructions
            result = self._client().complete(prompt, model=self.model_name)
        else:
            result = self._client().respond(
                prompt,
                model=self.model_name,
                instructions=instructions,
                tools=self.tools,
            )
        if result is None:
            raise RuntimeError(
                f"The {self.api_mode} transport returned no usable response."
            )
        return result

    async def ainvoke(self, messages: Iterable[BaseMessage]) -> AIMessage:
        result = await asyncio.to_thread(self._invoke, messages)
        return AIMessage(content=result.text, tool_calls=result.tool_calls or [])

    def _stream_events(
        self, messages: Iterable[BaseMessage]
    ) -> Iterable[dict[str, Any]]:
        instructions, prompt = _flatten(messages)
        client = self._client()
        if not isinstance(client, ResponsesClient):
            raise RuntimeError("Streaming requires the responses transport.")
        return client.stream_respond(
            prompt,
            model=self.model_name,
            instructions=instructions,
            tools=self.tools,
        )

    async def astream(
        self, messages: Iterable[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        materialized = list(messages)
        if self.api_mode == "completions":
            result = await asyncio.to_thread(self._invoke, materialized)
            yield AIMessageChunk(content=result.text)
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def produce() -> None:
            try:
                for event in self._stream_events(materialized):
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        task = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError(
                        f"The responses stream failed ({type(payload).__name__})."
                    ) from payload
                event = payload or {}
                event_type = event.get("type")
                if event_type == "text_delta":
                    yield AIMessageChunk(content=str(event.get("text") or ""))
                elif event_type == "tool_call_start":
                    yield AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": event.get("name") or "",
                                "args": "",
                                "id": event.get("id"),
                                "index": event.get("index", 0),
                            }
                        ],
                    )
                elif event_type == "tool_call_delta":
                    yield AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": None,
                                "args": event.get("arguments") or "",
                                "id": event.get("id"),
                                "index": event.get("index", 0),
                            }
                        ],
                    )
                elif event_type == "final":
                    result = event.get("result")
                    usage = getattr(result, "usage", None) or {}
                    yield AIMessageChunk(
                        content="",
                        response_metadata={
                            "model": getattr(result, "model", None),
                        },
                        usage_metadata={
                            "input_tokens": int(usage.get("input_tokens", 0) or 0),
                            "output_tokens": int(usage.get("output_tokens", 0) or 0),
                            "total_tokens": int(
                                usage.get("total_tokens", 0)
                                or usage.get("input_tokens", 0)
                                + usage.get("output_tokens", 0)
                            ),
                        },
                    )
        finally:
            await task


__all__ = ["TransportChatModel"]
