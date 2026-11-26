"""Unit tests for optional llama.cpp native endpoint contracts."""

import httpx

from providers.native import NativeClient, detect_native


class FakeResponse:
    def __init__(self, status_code=200, body=None, text="", error=None):
        self.status_code = status_code
        self.body = body
        self.text = text
        self.error = error

    def json(self):
        if self.error is not None:
            raise self.error
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("GET", "http://x"), response=self
            )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        return self.responses.pop(0)

    def get(self, url):
        self.calls.append(("GET", url, None))
        return self.responses.pop(0)

    def close(self):
        pass


def _client(responses):
    return NativeClient(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: FakeClient(responses),
    )


def test_health_ready_loading_and_absent():
    assert _client([FakeResponse(200)]).health() == "ready"
    assert _client([FakeResponse(503)]).health() == "loading"
    assert _client([FakeResponse(404)]).health() == "absent"


def test_props_returns_mapping_and_marks_detected():
    adapter = _client([FakeResponse(body={"n_ctx": 4096})])

    assert adapter.props() == {"n_ctx": 4096}
    assert adapter.profile.detected is True


def test_props_rejects_non_mapping():
    assert _client([FakeResponse(body=[])]).props() is None


def test_tokenize_accepts_plain_and_piece_tokens():
    plain = _client([FakeResponse(body={"tokens": [1, 2, 3]})])
    assert plain.tokenize("abc") == [1, 2, 3]

    pieces = _client([FakeResponse(body={"tokens": [[1, "a"], [2, "b"]]})])
    assert pieces.tokenize("ab", with_pieces=True) == [1, 2]


def test_tokenize_rejects_invalid_tokens():
    assert _client([FakeResponse(body={"tokens": ["x"]})]).tokenize("x") is None


def test_detokenize_returns_content():
    adapter = _client([FakeResponse(body={"content": "hello"})])

    assert adapter.detokenize([1, 2]) == "hello"


def test_apply_template_accepts_prompt_mapping_and_raw_string():
    mapping = _client([FakeResponse(body={"prompt": "<prompt>"})])
    assert mapping.apply_template(prompt="hi") == "<prompt>"

    raw = _client([FakeResponse(body="<raw>")])
    assert raw.apply_template(prompt="hi") == "<raw>"


def test_infill_sends_prefix_suffix_and_returns_content():
    adapter = _client([FakeResponse(body={"content": "middle"})])

    result = adapter.infill("a", "b", n_predict=4, stop=["x"], id_slot=1)

    assert result == "middle"


def test_slots_accepts_list_and_object():
    listed = _client([FakeResponse(body=[{"id": 0}])])
    assert listed.slots() == [{"id": 0}]

    wrapped = _client([FakeResponse(body={"slots": [{"id": 1}]})])
    assert wrapped.slots() == [{"id": 1}]


def test_metrics_only_returns_whitelisted_values():
    text = "\n".join(
        [
            "# HELP tokens",
            "llamacpp:tokens_predicted_total 42",
            "llamacpp:unknown_metric 99",
            "bad line",
        ]
    )
    adapter = _client([FakeResponse(text=text)])

    assert adapter.metrics() == {"llamacpp:tokens_predicted_total": 42.0}


def test_lora_adapters_normalizes_list_and_object():
    for body in (
        [{"id": 0, "path": "a", "scale": 1.0}],
        {"adapters": [{"id": 0, "path": "a", "scale": 1.0}]},
    ):
        adapter = _client([FakeResponse(body=body)])
        assert adapter.lora_adapters() == [{"id": 0, "path": "a", "scale": 1.0}]


def test_set_lora_posts_id_and_scale():
    client = FakeClient([FakeResponse(body={"ok": True})])
    adapter = NativeClient(
        "http://127.0.0.1:8080", client_factory=lambda **kwargs: client
    )

    assert adapter.set_lora(2, 0.5) is True
    assert client.calls[0][2] == {"id": 2, "scale": 0.5}


def test_unsupported_route_is_latched_after_404():
    client = FakeClient(
        [
            FakeResponse(404),
            FakeResponse(body={"tokens": [1]}),
        ]
    )
    adapter = NativeClient(
        "http://127.0.0.1:8080", client_factory=lambda **kwargs: client
    )

    assert adapter.tokenize("a") is None
    assert adapter.tokenize("a") is None
    assert len(client.calls) == 1
    assert "tokenize" in adapter.profile.unsupported


def test_network_error_returns_none_without_raising():
    class ExplodingClient(FakeClient):
        def request(self, method, url, json=None):
            raise httpx.ConnectError("down")

    adapter = NativeClient(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: ExplodingClient([]),
    )

    assert adapter.props() is None
    assert adapter.tokenize("a") is None
    assert adapter.slots() is None
    assert adapter.metrics() == {}
    assert adapter.lora_adapters() is None


def test_detect_native_retries_loading_then_reads_props():
    client = FakeClient(
        [
            FakeResponse(503),
            FakeResponse(200),
            FakeResponse(body={"n_ctx": 8192}),
        ]
    )

    profile = detect_native(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: client,
        sleep=lambda seconds: None,
    )

    assert profile.detected is True
    assert profile.ready is True
    assert profile.props == {"n_ctx": 8192}


def test_detect_native_reports_absent_without_props():
    client = FakeClient([FakeResponse(404)])

    profile = detect_native(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: client,
        sleep=lambda seconds: None,
    )

    assert profile.detected is False
    assert profile.props == {}


def test_detect_native_stops_after_loading_retries():
    client = FakeClient([FakeResponse(503), FakeResponse(503), FakeResponse(503)])

    profile = detect_native(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: client,
        retries=1,
        sleep=lambda seconds: None,
    )

    assert profile.detected is True
    assert profile.loading is True


def test_disabled_client_makes_no_calls():
    client = FakeClient([])
    adapter = NativeClient(
        "http://127.0.0.1:8080",
        client_factory=lambda **kwargs: client,
        enabled=False,
    )

    assert adapter.health() == "absent"
    assert adapter.props() is None
    assert client.calls == []
