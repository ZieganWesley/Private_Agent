"""Unit tests for the model catalog and its capability-detection wiring."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from agent.model_capabilities import (
    capabilities_with_limits,
    detect_openai_compatible_model_record,
)
from agent.model_catalog import (
    DEFAULT_CATALOG_TTL_SECONDS,
    PROVENANCE_DECLARED,
    PROVENANCE_DEFAULT,
    PROVENANCE_PROBE,
    ROLE_CHAT,
    ROLE_EMBEDDING,
    ROLE_RERANKER,
    ModelCatalog,
    build_model_record,
    classify_role,
    default_model_record,
    record_from_listing_entry,
)

BASE_URL = "http://127.0.0.1:1234/v1"


def _record(model_id: str = "m", context: int = 4096, ttl: float = 60.0, now=0.0):
    return build_model_record(
        model_id,
        base_url=BASE_URL,
        context_candidates=((PROVENANCE_PROBE, context),),
        ttl_seconds=ttl,
        now=now,
    )


def test_record_keeps_limits_modalities_and_provenance():
    record = build_model_record(
        "m",
        base_url=BASE_URL,
        endpoint_name="local",
        role_entries=({"type": "llm"},),
        context_candidates=(
            (PROVENANCE_DECLARED, 32000),
            (PROVENANCE_PROBE, 64000),
        ),
        input_candidates=((PROVENANCE_PROBE, 24000),),
        modality_candidates=((PROVENANCE_PROBE, ("text", "image")),),
    )

    assert record.role == ROLE_CHAT
    assert record.provenance_for("role") == PROVENANCE_PROBE
    assert record.context_length == 32000
    assert record.provenance_for("context_length") == PROVENANCE_DECLARED
    assert record.input_tokens == 24000
    assert record.provenance_for("input_tokens") == PROVENANCE_PROBE
    assert record.output_tokens is None
    assert record.provenance_for("output_tokens") == PROVENANCE_DEFAULT
    assert record.modalities == ("text", "image")
    assert record.endpoint_name == "local"


def test_default_record_marks_guessed_role_and_modality():
    record = default_model_record("mystery-chat", base_url=BASE_URL)

    assert record.role == ROLE_CHAT
    assert record.provenance_for("role") == PROVENANCE_DEFAULT
    assert record.context_length is None
    assert record.modalities == ("text",)
    assert record.provenance_for("modalities") == PROVENANCE_DEFAULT


@pytest.mark.parametrize(
    ("model_id", "entry", "expected_role", "provenance"),
    [
        ("plain-name", {"type": "embedding"}, ROLE_EMBEDDING, PROVENANCE_PROBE),
        ("plain-name", {"type": "reranker"}, ROLE_RERANKER, PROVENANCE_PROBE),
        ("text-embedding-3-small", None, ROLE_EMBEDDING, PROVENANCE_DEFAULT),
        ("bge-reranker-v2-m3", None, ROLE_RERANKER, PROVENANCE_DEFAULT),
        ("qwen2.5-7b", None, ROLE_CHAT, PROVENANCE_DEFAULT),
    ],
)
def test_classification_prefers_metadata_then_marks_name_guesses(
    model_id, entry, expected_role, provenance
):
    entries = (entry,) if entry else ()
    role, role_provenance = classify_role(model_id, entries)
    assert role == expected_role
    assert role_provenance == provenance


def test_listing_metadata_wins_over_an_embedding_like_name():
    record = record_from_listing_entry(
        {
            "id": "custom-embedding-name",
            "type": "llm",
            "context_length": 8192,
            "input_token_limit": 6000,
            "output_token_limit": 2048,
            "architecture": {"input_modalities": ["text"]},
        },
        base_url=BASE_URL,
    )

    assert record.role == ROLE_CHAT
    assert record.provenance_for("role") == PROVENANCE_PROBE
    assert record.context_length == 8192
    assert record.input_tokens == 6000
    assert record.output_tokens == 2048


def test_effective_limits_clamp_to_configured_ceilings_and_context():
    record = build_model_record(
        "m",
        base_url=BASE_URL,
        context_candidates=((PROVENANCE_PROBE, 16000),),
        input_candidates=((PROVENANCE_PROBE, 12000),),
        output_candidates=((PROVENANCE_PROBE, 8000),),
    )
    limits = record.effective_limits(ceiling_context=64000, ceiling_output=32000)
    assert limits.context == 16000
    assert limits.input_tokens == 12000
    assert limits.output_tokens == 8000

    tight = record.effective_limits(ceiling_context=4000, ceiling_output=32000)
    assert tight.context == 4000
    assert tight.input_tokens == 4000
    assert tight.output_tokens == 4000


def test_expired_record_is_refreshed_lazily_on_lookup():
    now = [100.0]
    catalog = ModelCatalog(ttl_seconds=10, clock=lambda: now[0])
    catalog.put(_record(context=4096, ttl=10, now=now[0]))
    calls = []

    fresh = catalog.lookup("m", BASE_URL, probe=lambda: calls.append("probe"))
    assert fresh.context_length == 4096
    assert calls == []

    now[0] += 11
    refreshed = catalog.lookup(
        "m",
        BASE_URL,
        probe=lambda: (
            calls.append("probe") or _record(context=8192, ttl=10, now=now[0])
        ),
    )
    assert refreshed.context_length == 8192
    assert calls == ["probe"]


def test_probe_failure_returns_the_stale_record():
    now = [100.0]
    catalog = ModelCatalog(ttl_seconds=10, clock=lambda: now[0])
    catalog.put(_record(context=4096, ttl=10, now=now[0]))
    now[0] += 11

    def broken_probe():
        raise OSError("endpoint down")

    stale = catalog.lookup("m", BASE_URL, probe=broken_probe)
    assert stale is not None
    assert stale.context_length == 4096


def test_single_flight_runs_one_probe_for_concurrent_same_key_lookups():
    catalog = ModelCatalog(ttl_seconds=60)
    started = threading.Event()
    release = threading.Event()
    counter = {"probes": 0}
    lock = threading.Lock()
    results = []

    def probe():
        with lock:
            counter["probes"] += 1
        started.set()
        assert release.wait(2), "probe was never released"
        return _record(context=4096, ttl=60)

    def lookup():
        results.append(catalog.lookup("m", BASE_URL, probe=probe))

    first = threading.Thread(target=lookup)
    first.start()
    assert started.wait(2)
    others = [threading.Thread(target=lookup) for _ in range(6)]
    for thread in others:
        thread.start()
    time.sleep(0.05)
    release.set()
    first.join(2)
    for thread in others:
        thread.join(2)

    assert counter["probes"] == 1
    assert len(results) == 7
    assert {result.context_length for result in results} == {4096}


def test_inflight_probe_for_one_key_does_not_block_another_key():
    catalog = ModelCatalog(ttl_seconds=60)
    blocked = threading.Event()
    release = threading.Event()
    results = []

    def slow_probe():
        blocked.set()
        assert release.wait(2), "slow probe was never released"
        return _record("slow", context=2048, ttl=60)

    def fast_probe():
        return _record("fast", context=1024, ttl=60)

    thread = threading.Thread(
        target=lambda: results.append(
            catalog.lookup("slow", BASE_URL, probe=slow_probe)
        )
    )
    thread.start()
    assert blocked.wait(2)

    started = time.monotonic()
    fast = catalog.lookup("fast", BASE_URL, probe=fast_probe)
    elapsed = time.monotonic() - started

    release.set()
    thread.join(2)

    assert fast is not None and fast.model_id == "fast"
    assert fast.context_length == 1024
    assert elapsed < 1.0, "an unrelated key must not wait for another key's probe"


def test_detection_record_exposes_limits_modalities_and_role():
    listing = {
        "data": [
            {
                "id": "m",
                "type": "llm",
                "context_length": 32768,
                "input_token_limit": 24000,
                "output_token_limit": 4096,
                "architecture": {"input_modalities": ["text", "image"]},
            }
        ]
    }

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: listing)

    record = detect_openai_compatible_model_record(
        "m",
        base_url="https://models.example.test/v1",
        client_factory=lambda **_kwargs: FakeClient(),
        catalog=ModelCatalog(ttl_seconds=DEFAULT_CATALOG_TTL_SECONDS),
    )

    surface = capabilities_with_limits(record)
    assert record.role == ROLE_CHAT
    assert record.context_length == 32768
    assert record.input_tokens == 24000
    assert record.output_tokens == 4096
    assert record.modalities == ("text", "image")
    assert surface["vision"] is True
    assert surface["role"] == ROLE_CHAT
    assert surface["input_tokens"] == 24000
    assert surface["modalities"] == ["text", "image"]
    assert surface["provenance"]["context_length"] == PROVENANCE_PROBE


def test_detection_uses_declared_overrides_for_cached_records():
    catalog = ModelCatalog(ttl_seconds=60)
    catalog.put(_record(context=4096, ttl=60, now=time.monotonic()))

    record = detect_openai_compatible_model_record(
        "m",
        base_url=BASE_URL,
        client_factory=lambda **_kwargs: pytest.fail("cached lookup must not probe"),
        catalog=catalog,
        overrides={"context_window": 2048, "tools": False},
    )

    assert record.context_length == 2048
    assert record.provenance_for("context_length") == PROVENANCE_DECLARED
    assert record.flags["tools"] is False
    assert record.provenance_for("tools") == PROVENANCE_DECLARED
