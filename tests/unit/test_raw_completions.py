"""P4: raw /v1/completions gating and its real utility call site."""

import asyncio
import types

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agent import session_summary
from config.settings import Settings
from database import PersistentMemory
from providers import utility
from providers.transports import TransportResult


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

    def post(self, url, json):
        self.calls.append((url, json))
        return self.responses.pop(0)

    def close(self):
        pass


class FakeLLM:
    def __init__(self, text="chat fallback", base_url="http://127.0.0.1:1234/v1"):
        self.text = text
        self.openai_api_base = base_url
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return types.SimpleNamespace(content=self.text)


def _messages():
    return [SystemMessage(content="be brief"), HumanMessage(content="hello")]


def test_flag_defaults_off_and_reads_models_section():
    assert Settings.from_config({}).raw_completions_enabled is False
    assert (
        Settings.from_config(
            {"models": {"raw_completions_enabled": True}}
        ).raw_completions_enabled
        is True
    )
    assert (
        Settings.from_config(
            {"models": {"raw_completions_enabled": "yes"}}
        ).raw_completions_enabled
        is True
    )


def test_enabled_raw_completions_uses_the_endpoint_once(monkeypatch):
    monkeypatch.setattr(utility, "raw_completions_enabled", lambda: True)
    client = FakeClient([FakeResponse({"choices": [{"text": "raw summary"}]})])
    llm = FakeLLM()

    text, used_raw = asyncio.run(
        utility.complete_utility_text(
            _messages(),
            model="m",
            chat_fallback=lambda: llm.ainvoke(_messages()),
            llm=llm,
            client_factory=lambda **kwargs: client,
        )
    )

    assert (text, used_raw) == ("raw summary", True)
    assert len(client.calls) == 1
    assert client.calls[0][0].endswith("/v1/completions")
    assert client.calls[0][1]["prompt"].startswith("System: be brief")
    assert llm.calls == 0


def test_disabled_raw_completions_never_requests_and_uses_chat(monkeypatch):
    monkeypatch.setattr(utility, "raw_completions_enabled", lambda: False)
    client = FakeClient([])
    llm = FakeLLM("chat summary")

    text, used_raw = asyncio.run(
        utility.complete_utility_text(
            _messages(),
            model="m",
            chat_fallback=lambda: llm.ainvoke(_messages()),
            llm=llm,
            client_factory=lambda **kwargs: client,
        )
    )

    assert (text, used_raw) == ("chat summary", False)
    assert client.calls == []
    assert llm.calls == 1


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, 500),
        FakeResponse({"choices": []}),
        FakeResponse({"choices": [{"text": 3}]}),
    ],
)
def test_transport_failure_or_bad_shape_degrades_to_chat(monkeypatch, response):
    monkeypatch.setattr(utility, "raw_completions_enabled", lambda: True)
    client = FakeClient([response])
    llm = FakeLLM("fallback answer")

    text, used_raw = asyncio.run(
        utility.complete_utility_text(
            _messages(),
            model="m",
            chat_fallback=lambda: llm.ainvoke(_messages()),
            llm=llm,
            client_factory=lambda **kwargs: client,
        )
    )

    assert (text, used_raw) == ("fallback answer", False)
    assert len(client.calls) == 1
    assert llm.calls == 1


def test_session_summary_uses_raw_completion_when_available(tmp_path, monkeypatch):
    import io

    from rich.console import Console

    monkeypatch.setattr(
        utility,
        "complete_utility",
        lambda messages, **kwargs: TransportResult(text="raw session summary"),
    )
    memory = PersistentMemory(db_path=str(tmp_path / "memory.sqlite"))
    llm = FakeLLM("chat session summary")

    saved = asyncio.run(
        session_summary._summarize_and_save_session(
            Console(file=io.StringIO(), width=200),
            memory,
            llm,
            "s1",
            [HumanMessage(content="hello")],
            "local",
        )
    )

    assert saved is True
    assert memory.get_all_episodic_summaries(session_id="s1") == ["raw session summary"]
    assert llm.calls == 0
