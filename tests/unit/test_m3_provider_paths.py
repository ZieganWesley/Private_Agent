"""M3: provider milestone over stubbed HTTP boundaries.

Covers native /completion, /apply-template, /infill, the P4 raw completions
utility path, and the P5b Responses tool + streaming transport.
"""

import asyncio
import json

import httpx
from langchain_core.messages import HumanMessage

from providers import utility
from providers.native import NativeClient
from providers.transport_model import TransportChatModel


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

    def iter_lines(self):
        for event in self.body.get("events", []) if isinstance(self.body, dict) else []:
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
    def __init__(self, responses, stream_response=None):
        self.responses = list(responses)
        self.stream_response = stream_response
        self.calls = []

    def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        return self.responses.pop(0)

    def post(self, url, json=None):
        self.calls.append(("POST", url, json))
        return self.responses.pop(0)

    def stream(self, method, url, json=None):
        self.calls.append((method, url, json))
        return StreamContext(self.stream_response)

    def close(self):
        pass


def _native(response, client=None):
    client = client or FakeClient([FakeResponse(response)])
    return NativeClient(
        "http://127.0.0.1:8080", client_factory=lambda **kwargs: client
    ), client


def test_native_completion_whitelists_options_and_parses_content():
    adapter, client = _native({"content": "native answer"})

    result = adapter.complete(
        "prompt",
        n_predict=16,
        temperature=0.2,
        top_k=40,
        stop=["STOP"],
        seed=7,
        return_tokens=True,
    )

    assert result == "native answer"
    method, url, payload = client.calls[0]
    assert (method, url) == ("POST", "http://127.0.0.1:8080/completion")
    assert payload == {
        "prompt": "prompt",
        "n_predict": 16,
        "temperature": 0.2,
        "top_p": 0.95,
        "cache_prompt": True,
        "stream": False,
        "return_tokens": True,
        "top_k": 40,
        "stop": ["STOP"],
        "seed": 7,
    }


def test_native_completion_latches_off_and_returns_none_on_404():
    adapter, client = _native({}, client=FakeClient([FakeResponse({}, 404)]))

    assert adapter.complete("prompt") is None
    assert "completion" in adapter.profile.unsupported
    assert adapter.complete("again") is None
    assert len(client.calls) == 1


def test_apply_template_and_infill_use_the_native_routes():
    client = FakeClient(
        [
            FakeResponse({"prompt": "rendered template"}),
            FakeResponse({"content": "filled middle"}),
        ]
    )
    adapter = NativeClient(
        "http://127.0.0.1:8080", client_factory=lambda **kwargs: client
    )

    prompt = adapter.apply_template(
        messages=[{"role": "user", "content": "hi"}], tools=[]
    )
    filled = adapter.infill("before ", " after", n_predict=8, temperature=0.1)

    assert prompt == "rendered template"
    assert filled == "filled middle"
    assert client.calls[0][1].endswith("/apply-template")
    assert client.calls[0][2]["messages"] == [{"role": "user", "content": "hi"}]
    assert client.calls[1][1].endswith("/infill")
    assert client.calls[1][2]["input_prefix"] == "before "


def test_raw_completions_utility_path_is_reachable_through_its_public_entry(monkeypatch):
    monkeypatch.setattr(utility, "raw_completions_enabled", lambda: True)
    client = FakeClient([FakeResponse({"choices": [{"text": "utility title"}]})])

    text, used_raw = asyncio.run(
        utility.complete_utility_text(
            [HumanMessage(content="summarize")],
            model="m",
            chat_fallback=lambda: "chat fallback",
            base_url="http://127.0.0.1:1234/v1",
            client_factory=lambda **kwargs: client,
        )
    )

    assert (text, used_raw) == ("utility title", True)
    assert client.calls[0][1].endswith("/completions")


def test_responses_tool_call_and_stream_are_usable_through_the_model():
    body = {
        "events": [
            {"type": "response.output_text.delta", "delta": "checking"},
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "fc", "name": "lookup"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc",
                "delta": '{"q":"x"}',
            },
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {"type": "function_call", "id": "fc", "name": "lookup", "arguments": '{"q":"x"}'}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
            },
        ]
    }
    client = FakeClient([], stream_response=FakeResponse(body))
    model = TransportChatModel(
        "http://127.0.0.1:8080/v1",
        "m",
        api_mode="responses",
        client_factory=lambda **kwargs: client,
    ).bind_tools([{"name": "lookup", "parameters": {}}])

    async def collect():
        return [chunk async for chunk in model.astream([HumanMessage(content="hi")])]

    chunks = asyncio.run(collect())

    assert "".join(chunk.content for chunk in chunks) == "checking"
    args = "".join(
        part["args"] for chunk in chunks for part in chunk.tool_call_chunks
    )
    assert json.loads(args) == {"q": "x"}
    assert client.calls[0][2]["stream"] is True
    assert client.calls[0][2]["tools"][0]["name"] == "lookup"
