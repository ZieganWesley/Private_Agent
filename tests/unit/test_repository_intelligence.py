"""Tests for repository intelligence and related Part M helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from private_agent.repository_intelligence import (
    AgentMetrics,
    ArchitectureMemory,
    build_repository_map,
    classify_memory,
    find_callees,
    find_callers,
    find_definition,
    find_dependencies,
    find_references,
    find_symbol,
    rank_memories,
    trace_execution_flow,
)
from private_agent.repository_intelligence.indexer import RepositoryIndex


def _sample_repo(root: Path) -> None:
    src = root / "src"
    src.mkdir()
    (src / "a.py").write_text(
        "import os\n\n\ndef foo():\n    bar()\n\n\nclass Baz(Qux):\n    def method(self):\n        pass\n",
        encoding="utf-8",
    )
    (root / "main.js").write_text(
        "function greet(name) { return name; }\nmodule.exports = greet;\n",
        encoding="utf-8",
    )


def test_repository_index_extracts_symbols_and_edges():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _sample_repo(root)
        db = str(root / "index.db")
        index = RepositoryIndex(db_path=db, root_path=str(root))
        try:
            stats = index.index_repository(str(root))
            assert stats["files_indexed"] >= 1
            assert stats["symbols_indexed"] >= 3
            symbols = index.find_symbols("foo")
            assert symbols and symbols[0]["kind"] == "function"
            assert index.callers_of("bar")
            assert index.callees_of("foo")
            classes = index.find_symbols("Baz")
            assert classes and classes[0]["kind"] == "class"
        finally:
            index.close()


def test_incremental_index_skips_unchanged_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _sample_repo(root)
        db = str(root / "index.db")
        index = RepositoryIndex(db_path=db, root_path=str(root))
        try:
            first = index.index_repository(str(root))
            second = index.index_repository(str(root), incremental=True)
            assert first["files_indexed"] >= 1
            assert second["files_skipped"] >= 1
        finally:
            index.close()


def test_query_helpers_return_results():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _sample_repo(root)
        db = str(root / "index.db")
        assert find_symbol("foo", root_path=str(root), db_path=db)
        assert find_definition("foo", root_path=str(root), db_path=db)["name"] == "foo"
        assert isinstance(find_references("bar", root_path=str(root), db_path=db), list)
        assert isinstance(find_callers("bar", root_path=str(root), db_path=db), list)
        assert isinstance(find_callees("foo", root_path=str(root), db_path=db), list)
        assert isinstance(find_dependencies("src/a.py", root_path=str(root), db_path=db), list)
        assert isinstance(trace_execution_flow("foo", root_path=str(root), db_path=db), list)


def test_repository_map_contains_symbols():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _sample_repo(root)
        db = str(root / "index.db")
        content = build_repository_map(root_path=str(root), db_path=db)
        assert "Repository map" in content
        assert "foo" in content


def test_architecture_memory_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "arch.db")
        mem = ArchitectureMemory(db_path=db)
        try:
            rid = mem.record_decision(
                root=".",
                title="Use SQLite",
                decision="Store local data in SQLite",
            )
            assert rid > 0
            mem.record_fact(root=".", kind="architecture", content="Layered design")
            results = mem.recall(root=".", query="SQLite")
            assert any(r["type"] == "decision" for r in results)
        finally:
            mem.close()


def test_agent_metrics_summary():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "metrics.db")
        metrics = AgentMetrics(db_path=db)
        try:
            metrics.record("tool", "read_file", value=2.0)
            summary = metrics.summary()
            assert summary["total_events"] == 1
            report = metrics.health_report()
            assert "tool/read_file" in report
        finally:
            metrics.close()


def test_memory_classification_and_ranking():
    assert classify_memory("remember this fact") in {
        "working",
        "long_term",
        "project",
        "architecture",
        "user",
        "episodic",
    }
    memories = [
        {"text": "python style guide", "created_at": 1, "importance": 0.9},
        {"text": "old note", "created_at": 0, "importance": 0.1},
    ]
    ranked = rank_memories(memories, query="python")
    assert ranked[0]["text"] == "python style guide"
    assert "_score" in ranked[0]


def test_parallel_parse_preserves_order():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(8):
            (root / f"f{i}.py").write_text(
                f"def foo{i}():\n    bar{i}()\n", encoding="utf-8"
            )
        db = str(root / "index.db")
        index = RepositoryIndex(db_path=db, root_path=str(root))
        try:
            pending = [
                (root / f"f{i}.py", f"f{i}.py", ".py") for i in range(8)
            ]
            results = index._parse_files_parallel(pending)
            assert len(results) == 8
            for i, (syms, _edges) in enumerate(results):
                assert syms and syms[0].name == f"foo{i}"
        finally:
            index.close()


def test_parallel_parse_falls_back_to_in_process(monkeypatch):
    import private_agent.repository_intelligence.indexer as indexer_module

    class _BrokenExecutor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("unavailable")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        indexer_module, "ProcessPoolExecutor", _BrokenExecutor
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(8):
            (root / f"g{i}.py").write_text(
                f"def baz{i}():\n    qux{i}()\n", encoding="utf-8"
            )
        db = str(root / "index.db")
        index = RepositoryIndex(db_path=db, root_path=str(root))
        try:
            pending = [
                (root / f"g{i}.py", f"g{i}.py", ".py") for i in range(8)
            ]
            results = index._parse_files_parallel(pending)
            assert len(results) == 8
            for i, (syms, _edges) in enumerate(results):
                assert syms and syms[0].name == f"baz{i}"
        finally:
            index.close()
