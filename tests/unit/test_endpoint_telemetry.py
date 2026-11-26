"""P6: endpoint telemetry is recorded on real request paths and inspectable."""

import httpx
import pytest

from agent import provider_report
from providers.endpoint_telemetry import (
    endpoint_snapshot,
    format_endpoint_report,
    reset_endpoint_telemetry,
)
from providers.native import NativeClient
from providers.transports import CompletionsClient, ResponsesClient


@pytest.fixture(autouse=True)
def _reset():
    reset_endpoint_telemetry()
    provider_report.reset_provider_report()
    yield
    reset_endpoint_telemetry()
    provider_report.reset_provider_report()


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "failed", request=httpx.Request("POST", "http://test"), response=self
            )

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def post(self, url, json=None):
        if self.error is not None:
            raise self.error
        return self.response

    def get(self, url):
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        pass


def _completions(response=None, error=None, base_url="http://127.0.0.1:1234/v1"):
    return CompletionsClient(
        base_url,
        client_factory=lambda **kwargs: FakeClient(response, error),
    )


def test_completions_request_records_counts_latency_and_usage():
    response = FakeResponse(
        {
            "choices": [{"text": "hello"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }
    )

    result = _completions(response).complete("PROMPT TEXT", model="m")

    assert result is not None and result.text == "hello"
    rows = endpoint_snapshot()
    assert len(rows) == 1
    row = rows[0]
    assert (row["host"], row["operation"]) == ("127.0.0.1:1234", "completions")
    assert (row["requests"], row["successes"], row["failures"]) == (1, 1, 0)
    assert row["input_tokens"] == 11
    assert row["output_tokens"] == 3
    assert row["avg_seconds"] >= 0
    assert sum(row["latency"].values()) == 1

    report = format_endpoint_report()
    assert "127.0.0.1:1234" in report
    assert "requests=1" in report
    assert "PROMPT TEXT" not in report


def test_transport_failure_is_counted_without_content():
    client = _completions(error=httpx.ConnectError("refused"))

    assert client.complete("SECRET PROMPT", model="m") is None

    row = endpoint_snapshot()[0]
    assert (row["requests"], row["successes"], row["failures"]) == (1, 0, 1)
    assert row["last_error_type"] == "ConnectError"
    assert "SECRET PROMPT" not in format_endpoint_report()


def test_malformed_response_counts_as_failure():
    response = FakeResponse({"choices": []})

    assert _completions(response).complete("prompt", model="m") is None

    row = endpoint_snapshot()[0]
    assert row["failures"] == 1
    assert row["last_error_type"] == "invalid_response"


def test_responses_request_records_under_its_operation():
    response = FakeResponse(
        {
            "output": [{"content": [{"text": "answer"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
    )
    client = ResponsesClient(
        "http://localhost:8080/v1",
        client_factory=lambda **kwargs: FakeClient(response),
    )

    assert client.respond("prompt", model="m").text == "answer"

    row = endpoint_snapshot()[0]
    assert row["operation"] == "responses"
    assert (row["input_tokens"], row["output_tokens"]) == (5, 2)


def test_native_health_is_recorded_through_the_client_path():
    response = FakeResponse({}, status_code=503)
    client = NativeClient(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: FakeClient(response),
    )

    assert client.health() == "loading"

    row = endpoint_snapshot()[0]
    assert (row["host"], row["operation"]) == ("127.0.0.1:8080", "health")
    assert row["failures"] == 1
    assert row["last_error_type"] == "not_ready"


def test_provider_report_combines_embedding_route_and_endpoint_usage():
    provider_report.record_embedding_route(
        "openai-compatible", "bge-small", "http://127.0.0.1:1234/v1"
    )
    response = FakeResponse({"choices": [{"text": "ok"}]})
    _completions(response).complete("prompt", model="m")

    report = provider_report.format_provider_report()

    assert "embeddings: provider=openai-compatible" in report
    assert "model=bge-small" in report
    assert "base_url=http://127.0.0.1:1234/v1" in report
    assert "requests=1" in report
    assert "prompt" not in report.split("Endpoint usage:")[1].lower()


def test_provider_report_without_requests_is_explicit():
    report = provider_report.format_provider_report()

    assert "embeddings not resolved yet" in report
    assert "no requests recorded" in report
