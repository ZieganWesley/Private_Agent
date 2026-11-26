"""Unit tests for optional /v1/completions and /v1/responses clients."""

import asyncio

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from agent import model_selection
from providers.transport_model import TransportChatModel
from providers.transports import CompletionsClient, ResponsesClient


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


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, json):
        self.calls.append((url, json))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def _factory(client):
    return lambda **kwargs: client


def test_completions_returns_text_and_payload():
    client = FakeClient(
        [
            FakeResponse(
                {
                    "choices": [{"text": "hello"}],
                    "model": "m",
                    "usage": {"total_tokens": 2},
                }
            )
        ]
    )
    adapter = CompletionsClient(
        "http://127.0.0.1:8080/v1", client_factory=_factory(client)
    )

    result = adapter.complete("prompt", model="m", max_tokens=8, stop=["x"])

    assert result.text == "hello"
    assert result.usage == {"total_tokens": 2}
    assert client.calls[0][0] == "http://127.0.0.1:8080/v1/completions"
    assert client.calls[0][1]["prompt"] == "prompt"
    assert client.calls[0][1]["stop"] == ["x"]


def test_completions_rejects_invalid_shape():
    adapter = CompletionsClient(
        "http://127.0.0.1:8080/v1",
        client_factory=_factory(FakeClient([FakeResponse({"choices": []})])),
    )

    assert adapter.complete("prompt", model="m") is None


def test_completions_returns_none_on_http_error():
    adapter = CompletionsClient(
        "http://127.0.0.1:8080/v1",
        client_factory=_factory(FakeClient([FakeResponse({}, 500)])),
    )

    assert adapter.complete("prompt", model="m") is None


def test_disabled_completions_makes_no_request():
    client = FakeClient([])
    adapter = CompletionsClient(
        "http://127.0.0.1:8080/v1",
        client_factory=_factory(client),
        enabled=False,
    )

    assert adapter.complete("prompt", model="m") is None
    assert client.calls == []


def test_responses_reads_output_text():
    client = FakeClient([FakeResponse({"output_text": "answer", "model": "m"})])
    adapter = ResponsesClient(
        "http://127.0.0.1:8080/v1", client_factory=_factory(client)
    )

    result = adapter.respond("question", model="m", instructions="be brief")

    assert result.text == "answer"
    assert client.calls[0][0] == "http://127.0.0.1:8080/v1/responses"
    assert client.calls[0][1]["instructions"] == "be brief"


def test_responses_reads_output_content_blocks():
    body = {
        "output": [{"content": [{"type": "output_text", "text": "a"}, {"text": "b"}]}]
    }
    adapter = ResponsesClient(
        "http://127.0.0.1:8080/v1",
        client_factory=_factory(FakeClient([FakeResponse(body)])),
    )

    assert adapter.respond("q", model="m").text == "ab"


def test_responses_rejects_invalid_shape():
    adapter = ResponsesClient(
        "http://127.0.0.1:8080/v1",
        client_factory=_factory(FakeClient([FakeResponse({"output": []})])),
    )

    assert adapter.respond("q", model="m") is None


def test_api_key_is_sent_as_bearer_header():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeClient([FakeResponse({"choices": [{"text": "x"}]})])

    adapter = CompletionsClient(
        "http://127.0.0.1:8080/v1",
        api_key="secret",
        client_factory=factory,
    )
    adapter.complete("p", model="m")

    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_transport_model_uses_completions_route():
    client = FakeClient([FakeResponse({"choices": [{"text": "hello"}]})])
    model = TransportChatModel(
        "http://127.0.0.1:8080/v1",
        "m",
        api_mode="completions",
        client_factory=_factory(client),
    )

    result = asyncio.run(
        model.ainvoke([SystemMessage(content="be brief"), HumanMessage(content="hi")])
    )

    assert result.content == "hello"
    assert client.calls[0][0].endswith("/v1/completions")
    assert "be brief" in client.calls[0][1]["prompt"]
    assert "HumanMessage: hi" in client.calls[0][1]["prompt"]


def test_transport_model_uses_responses_route_with_instructions():
    client = FakeClient([FakeResponse({"output_text": "answer"})])
    model = TransportChatModel(
        "http://127.0.0.1:8080/v1",
        "m",
        api_mode="responses",
        client_factory=_factory(client),
    )

    result = asyncio.run(model.ainvoke([SystemMessage(content="be brief")]))

    assert result.content == "answer"
    assert client.calls[0][0].endswith("/v1/responses")
    assert client.calls[0][1]["instructions"] == "be brief"


def test_selected_local_model_honors_configured_api_mode(monkeypatch):
    class Runtime:
        base_url = "http://127.0.0.1:8080/v1"

    monkeypatch.setattr(model_selection, "PROVIDER_API_MODE", "responses")
    result = model_selection._make_selected_local_chat_model(
        Runtime(),
        "m",
        thinking_enabled=False,
        thinking_effort="medium",
        capabilities={},
    )

    assert isinstance(result, TransportChatModel)
    assert result.api_mode == "responses"
