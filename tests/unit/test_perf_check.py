"""The performance guard fails on growth beyond 20% (J13.10)."""

import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "perf_check.py"
_SPEC = importlib.util.spec_from_file_location("perf_check", _SCRIPT)
assert _SPEC and _SPEC.loader
perf_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_check)


def test_growth_within_twenty_percent_passes():
    baseline = {"startup_import_seconds": 1.0, "rag_refresh_seconds": 0.2}
    current = {"startup_import_seconds": 1.19, "rag_refresh_seconds": 0.2}
    assert perf_check.compare(current, baseline) == []


def test_growth_beyond_twenty_percent_fails_for_each_budget():
    baseline = {"startup_import_seconds": 1.0, "rag_refresh_seconds": 0.2}
    current = {"startup_import_seconds": 1.25, "rag_refresh_seconds": 0.25}
    problems = perf_check.compare(current, baseline)
    assert len(problems) == 2
    assert "startup_import_seconds" in problems[0]
    assert "rag_refresh_seconds" in problems[1]


def test_smaller_timings_pass():
    baseline = {"startup_import_seconds": 1.0, "rag_refresh_seconds": 0.2}
    current = {"startup_import_seconds": 0.8, "rag_refresh_seconds": 0.01}
    assert perf_check.compare(current, baseline) == []
