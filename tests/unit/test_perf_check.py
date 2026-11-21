"""The performance guard compares calibrated ratios (J13.10)."""

import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "perf_check.py"
_SPEC = importlib.util.spec_from_file_location("perf_check", _SCRIPT)
assert _SPEC and _SPEC.loader
perf_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_check)


def test_growth_within_twenty_percent_passes():
    baseline = {"startup_import": 7.0, "rag_refresh": 0.6}
    current = {"startup_import": 8.3, "rag_refresh": 0.6}
    assert perf_check.compare(current, baseline) == []


def test_growth_beyond_twenty_percent_fails_for_each_budget():
    baseline = {"startup_import": 7.0, "rag_refresh": 0.6}
    current = {"startup_import": 8.5, "rag_refresh": 0.8}
    problems = perf_check.compare(current, baseline)
    assert len(problems) == 2
    assert "startup_import" in problems[0]
    assert "rag_refresh" in problems[1]


def test_smaller_ratios_pass():
    baseline = {"startup_import": 7.0, "rag_refresh": 0.6}
    current = {"startup_import": 5.0, "rag_refresh": 0.1}
    assert perf_check.compare(current, baseline) == []


def test_slower_hardware_keeps_the_same_ratio():
    # A machine twice as slow doubles both the measurement and the
    # calibration workload, so the ratio (and the check) is unchanged.
    value, calibration = 1.6, 0.25
    assert perf_check.ratio(value, calibration) == perf_check.ratio(
        value * 2, calibration * 2
    )


def test_ratio_rejects_a_zero_calibration():
    try:
        perf_check.ratio(1.0, 0)
    except ValueError as error:
        assert "calibration" in str(error)
    else:
        raise AssertionError("a zero calibration measurement must be rejected")
