import json

import pytest
from langchain_core.documents import Document

from scripts.benchmark_rag import (
    _benchmark_queries,
    _retrieval_metrics,
    _runtime_hardware_reports,
    load_labeled_queries,
)


def test_load_labeled_queries_validates_and_normalizes_source_paths(tmp_path):
    query_file = tmp_path / "queries.json"
    query_file.write_text(
        json.dumps(
            [
                {
                    "query": " Where is the offline guide? ",
                    "expected_sources": ["docs/guide.md", "docs/faq.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_labeled_queries(query_file) == [
        {
            "query": "Where is the offline guide?",
            "expected_sources": {"docs/guide.md", "docs/faq.md"},
        }
    ]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"query": "", "expected_sources": ["guide.md"]}, "nonempty query"),
        ({"query": "question", "expected_sources": []}, "expected_sources"),
        (
            {"query": "question", "expected_sources": ["../outside.md"]},
            "must be relative",
        ),
    ],
)
def test_load_labeled_queries_rejects_invalid_entries(tmp_path, entry, message):
    query_file = tmp_path / "queries.json"
    query_file.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_labeled_queries(query_file)


def test_retrieval_metrics_compute_recall_and_reciprocal_rank(tmp_path):
    documents = tmp_path / "docs"
    documents.mkdir()
    relevant_one = documents / "one.md"
    relevant_two = documents / "two.md"
    irrelevant = documents / "other.md"
    retrieved = [
        Document(page_content="irrelevant", metadata={"source": str(irrelevant)}),
        Document(page_content="first match", metadata={"source": str(relevant_one)}),
        Document(page_content="second match", metadata={"source": str(relevant_two)}),
    ]

    assert _retrieval_metrics(
        retrieved,
        {"one.md", "two.md", "missing.md"},
        documents,
    ) == pytest.approx((2 / 3, 1 / 2))


def test_benchmark_reports_recall_mrr_and_latency_statistics(tmp_path):
    documents = tmp_path / "docs"
    documents.mkdir()
    relevant = documents / "guide.md"

    class FakeRetriever:
        def similarity_search(self, query, k):
            assert query == "offline guide"
            assert k == 2
            return [
                Document(
                    page_content="guide",
                    metadata={"source": str(relevant)},
                )
            ]

    result = _benchmark_queries(
        FakeRetriever(),
        [{"query": "offline guide", "expected_sources": {"guide.md"}}],
        documents,
        top_k=2,
        repeats=2,
    )

    assert result["queries"] == 1
    assert result["repeats"] == 2
    assert result["top_k"] == 2
    assert result["mean_recall_at_k"] == 1
    assert result["mean_reciprocal_rank"] == 1
    assert result["median_latency_ms"] >= 0
    assert result["mean_latency_ms"] >= 0


def test_hardware_reports_cover_every_reachable_runtime(monkeypatch):
    import scripts.benchmark_rag as benchmark

    monkeypatch.setattr(
        benchmark,
        "_discover_available_local_runtimes",
        lambda: [
            benchmark_runtime("Ollama", "ollama", "http://127.0.0.1:11434"),
            benchmark_runtime(
                "LM Studio", "openai-compatible", "http://127.0.0.1:1234"
            ),
        ],
    )
    monkeypatch.setattr(
        benchmark, "inspect_ollama_hardware", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(benchmark, "format_hardware_status", lambda info: "OLLAMA")

    reports = _runtime_hardware_reports()

    assert [name for name, _report in reports] == ["Ollama", "LM Studio"]
    assert reports[0][1] == "OLLAMA"
    assert "standardized" in reports[1][1]


def test_hardware_reports_are_empty_without_local_providers(monkeypatch):
    import scripts.benchmark_rag as benchmark

    monkeypatch.setattr(benchmark, "_discover_available_local_runtimes", lambda: [])

    assert _runtime_hardware_reports() == []


def test_hardware_reports_survive_a_failing_probe(monkeypatch):
    import scripts.benchmark_rag as benchmark

    monkeypatch.setattr(
        benchmark,
        "_discover_available_local_runtimes",
        lambda: [benchmark_runtime("Ollama", "ollama", "http://127.0.0.1:11434")],
    )

    def exploding(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(benchmark, "inspect_ollama_hardware", exploding)

    reports = _runtime_hardware_reports()

    assert reports == [("Ollama", "Hardware status unavailable: connection refused")]


def benchmark_runtime(name, kind, base_url):
    from types import SimpleNamespace

    return SimpleNamespace(name=name, kind=kind, base_url=base_url, models=())
