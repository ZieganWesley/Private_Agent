"""P5b: Responses tool definitions, tool-call parsing, and streaming."""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from providers.responses_protocol import tool_definitions
from providers.transport_model import TransportChatModel
from providers.transports import ResponsesClient


class FakeResponse:
    def __init__(self, body=None, status_code=200):
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("POST", "http://x"), response=self
            )

    def json(self):
        return self.body


class FakeStreamResponse(FakeResponse):
    def __init__(self, events, status_code=200):
        super().__init__({}, status_code)
        self.events = events

    def iter_lines(self):
        for event in self.events:
            if isinstance(event, str):
                yield event
            else:
                yield f"data: {json.dumps(event)}"
        yield "data: [DONE]"


class StreamContext:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, *exc_info):
        return False


class FakeClient:
    def __init__(self, responses=(), stream_response=None):
        self.responses = list(responses)
        self.stream_response = stream_response
        self.calls = []
        self.stream_calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        return self.responses.pop(0)

    def stream(self, method, url, json):
        self.stream_calls.append((method, url, json))
        return StreamContext(self.stream_response)

    def close(self):
        pass


def _factory(client):
    return lambda **kwargs: client


def test_tool_definitions_cover_dicts_and_langchain_style_tools():
    class Tool:
        name = "lookup"
        description = "Look something up"
        args_schema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }

    definitions = tool_definitions(
        [Tool(), {"type": "function", "function": {"name": "second"}}]
    )

    assert definitions[0] == {
        "type": "function",
        "name": "lookup",
        "description": "Look something up",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        },
    }
    assert definitions[1]["name"] == "second"
    assert definitions[1]["parameters"] == {"type": "object", "properties": {}}


def test_tool_calls_round_trip_through_the_transport_model():
    body = {
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q": "x"}',
            }
        ]
    }
    client = FakeClient([FakeResponse(body)])
    model = TransportChatModel(
        "http://127.0.0.1:8080/v1",
        "m",
        api_mode="responses",
        client_factory=_factory(client),
    ).bind_tools([{"name": "lookup", "description": "find", "parameters": {}}])

    message = asyncio.run(
        model.ainvoke([SystemMessage(content="be brief"), HumanMessage(content="hi")])
    )

    assert isinstance(message, AIMessage)
    assert message.content == ""
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "lookup"
    assert message.tool_calls[0]["args"] == {"q": "x"}
    assert message.tool_calls[0]["id"] == "call_1"
    payload = client.calls[0][1]
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "find",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_usage_and_text_are_parsed_from_the_same_response():
    body = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
            {
                "type": "function_call",
                "id": "fc_1",
                "name": "lookup",
                "arguments": {"q": "y"},
            },
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "model": "m2",
    }
    client = FakeClient([FakeResponse(body)])
    adapter = ResponsesClient(
        "http://127.0.0.1:8080/v1", client_factory=_factory(client)
    )

    result = adapter.respond("q", model="m")

    assert result.text == "done"
    assert result.model == "m2"
    assert result.usage == {"input_tokens": 7, "output_tokens": 3}
    assert result.tool_calls == [{"name": "lookup", "args": {"q": "y"}, "id": "fc_1"}]


def test_streaming_yields_incremental_text_and_tool_arguments():
    events = [
        {"type": "response.output_text.delta", "delta": "Hel"},
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"q": ',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '"x"}',
        },
        {"type": "response.output_text.delta", "delta": "lo"},
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"q": "x"}',
                    },
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "model": "m",
            },
        },
    ]
    client = FakeClient(stream_response=FakeStreamResponse(events))
    model = TransportChatModel(
        "http://127.0.0.1:8080/v1",
        "m",
        api_mode="responses",
        client_factory=_factory(client),
    ).bind_tools([{"name": "lookup", "parameters": {}}])

    async def collect():
        return [chunk async for chunk in model.astream([HumanMessage(content="hi")])]

    chunks = asyncio.run(collect())

    assert "".join(chunk.content for chunk in chunks) == "Hello"
    tool_chunks = [c for chunk in chunks for c in chunk.tool_call_chunks]
    assert tool_chunks[0]["name"] == "lookup"
    arguments = "".join(c["args"] for c in tool_chunks)
    assert json.loads(arguments) == {"q": "x"}
    assert tool_chunks[0]["id"] == "call_1"
    assert chunks[-1].usage_metadata == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert client.stream_calls[0][2]["stream"] is True
    assert client.stream_calls[0][2]["tools"][0]["name"] == "lookup"


def test_stream_failure_surfaces_as_a_runtime_error():
    client = FakeClient(stream_response=FakeStreamResponse([], status_code=500))
    adapter = ResponsesClient(
        "http://127.0.0.1:8080/v1", client_factory=_factory(client)
    )

    with pytest.raises(RuntimeError):
        list(adapter.stream_respond("q", model="m"))
