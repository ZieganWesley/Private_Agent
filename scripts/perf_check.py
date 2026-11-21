"""Measure startup, RAG refresh and offline token counting (J13.10).

Wall-clock seconds differ between machines, so the check does not compare
seconds. Each measurement is reported relative to a calibration workload run
by the same interpreter on the same machine: a fixed pure-Python task for the
RAG refresh, and the same task in a fresh child process (which also pays
interpreter startup) for the startup import. Those ratios are hardware
independent, and the check fails when either grows by more than 20% against
``scripts/perf_baseline.json``.

The baseline stores the ratios it compares plus one ``reference_seconds``
block that is informational only, measured on the machine named in
``reference_hardware``; nothing compares against it.

Usage:
    python scripts/perf_check.py            # compare with the baseline
    python scripts/perf_check.py --update-baseline
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_PATH = pathlib.Path(__file__).with_name("perf_baseline.json")
GROWTH_LIMIT = 0.20
BASELINE_FORMAT = 2
CORPUS_FILES = 300
CORPUS_SECTIONS = 8
# Sized to take roughly a quarter second, so it is large compared with timer
# and interpreter noise but small compared with the startup import it scales.
CALIBRATION_ITERATIONS = 2_400_000
CALIBRATION_PARTS = 360_000

_STARTUP_CODE = "import private_agent.agent.runtime"

_CALIBRATION_CODE = """
import sys

sys.path.insert(0, {scripts_directory!r})
from perf_check import _calibration_workload

print(_calibration_workload())
"""

_OFFLINE_TOKEN_CODE = """
import socket

def _blocked(*args, **kwargs):
    raise OSError("network disabled for the performance check")

class _BlockedSocket(socket.socket):
    def connect(self, *args, **kwargs):
        _blocked()

socket.socket = _BlockedSocket
socket.create_connection = _blocked
from private_agent.agent.prompts import token_count

print(token_count("offline token counting " * 200))
"""


def _calibration_workload() -> int:
    """Fixed CPU-bound work with no I/O, imports or randomness."""
    total = 0
    for index in range(CALIBRATION_ITERATIONS):
        total += (index * index) % 977
    parts = [f"token-{index % 97}" for index in range(CALIBRATION_PARTS)]
    return total + len(" ".join(parts).split())


def _child_environment(home: pathlib.Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(pathlib.Path(__file__).parent)]
    )
    environment["HOME"] = str(home)
    environment["XDG_CONFIG_HOME"] = str(home / ".config")
    environment.pop("PYTHONHASHSEED", None)
    return environment


def measure_startup_import(runs: int = 3) -> float:
    """Return the median seconds spent importing the runtime in a child."""
    timings = []
    with tempfile.TemporaryDirectory() as home:
        environment = _child_environment(pathlib.Path(home))
        for _ in range(runs):
            started = time.perf_counter()
            subprocess.run(
                [sys.executable, "-c", _STARTUP_CODE],
                check=True,
                capture_output=True,
                env=environment,
                timeout=180,
            )
            timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def measure_process_calibration(runs: int = 3) -> float:
    """Return the median seconds for interpreter start plus calibration work."""
    timings = []
    expected = _calibration_workload()
    code = _CALIBRATION_CODE.format(
        scripts_directory=str(pathlib.Path(__file__).parent)
    )
    with tempfile.TemporaryDirectory() as home:
        environment = _child_environment(pathlib.Path(home))
        for _ in range(runs):
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                env=environment,
                text=True,
                timeout=180,
            )
            timings.append(time.perf_counter() - started)
            if completed.stdout.strip() != str(expected):
                raise RuntimeError(
                    "The calibration workload returned an unexpected result: "
                    f"{completed.stdout.strip()!r}"
                )
    return statistics.median(timings)


def measure_inprocess_calibration(runs: int = 3) -> float:
    """Return the median seconds for the calibration work in this process."""
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        _calibration_workload()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def _fixed_corpus(root: pathlib.Path) -> list[pathlib.Path]:
    files = []
    for index in range(CORPUS_FILES):
        body = "\n\n".join(
            f"## Section {section}\n\n"
            f"Paragraph {section} of document {index} explains the fixed "
            "corpus used by the performance check in plain prose so chunking "
            "work is stable between runs and machines. "
            * 3
            for section in range(CORPUS_SECTIONS)
        )
        path = root / f"doc-{index:03d}.md"
        path.write_text(f"# Document {index}\n\n{body}\n", encoding="utf-8")
        files.append(path)
    return files


def measure_rag_refresh(runs: int = 3) -> float:
    """Return the median seconds to read, chunk and index a fixed corpus."""
    from langchain_core.documents import Document

    from private_agent.rag import indexing
    from private_agent.rag.retrieval import HybridRAGRetriever

    timings = []
    with tempfile.TemporaryDirectory() as corpus_root:
        files = _fixed_corpus(pathlib.Path(corpus_root))
        for _ in range(runs):
            indexing.clear_chunk_cache()
            started = time.perf_counter()
            documents = []
            for path in files:
                content = path.read_text(encoding="utf-8")
                for chunk in indexing._split_into_chunks(content, extension=".md"):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": str(path), "chunk": len(documents)},
                        )
                    )
            HybridRAGRetriever(None, documents)
            timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def measure_offline_token_count(runs: int = 2) -> tuple[float, int]:
    """Count tokens in a child whose network access raises, then time it."""
    timings = []
    counted = 0
    with tempfile.TemporaryDirectory() as home:
        environment = _child_environment(pathlib.Path(home))
        for _ in range(runs):
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, "-c", _OFFLINE_TOKEN_CODE],
                check=True,
                capture_output=True,
                env=environment,
                text=True,
                timeout=120,
            )
            timings.append(time.perf_counter() - started)
            counted = int(completed.stdout.strip().splitlines()[-1])
    return statistics.median(timings), counted


def ratio(value: float, calibration: float) -> float:
    """Express a measurement in calibration units, guarding a zero divisor."""
    if calibration <= 0:
        raise ValueError("calibration measurement must be positive")
    return value / calibration


def compare(current: dict[str, float], baseline: dict[str, float]) -> list[str]:
    """Return human-readable regressions over ``GROWTH_LIMIT`` in ratio units."""
    problems = []
    for key in ("startup_import", "rag_refresh"):
        previous = baseline.get(key)
        now = current.get(key)
        if not previous or now is None:
            continue
        if now > previous * (1 + GROWTH_LIMIT):
            problems.append(
                f"{key} grew from {previous:.2f}x to {now:.2f}x its calibration "
                f"workload (limit {GROWTH_LIMIT:.0%})"
            )
    return problems


def collect_results() -> dict:
    """Measure everything once and return seconds, ratios and calibration."""
    startup_seconds = measure_startup_import()
    process_calibration = measure_process_calibration()
    refresh_seconds = measure_rag_refresh()
    cpu_calibration = measure_inprocess_calibration()
    token_seconds, counted_tokens = measure_offline_token_count()
    return {
        "ratios": {
            "startup_import": round(ratio(startup_seconds, process_calibration), 4),
            "rag_refresh": round(ratio(refresh_seconds, cpu_calibration), 4),
        },
        "calibration_seconds": {
            "child_process": round(process_calibration, 4),
            "in_process": round(cpu_calibration, 4),
        },
        "reference_seconds": {
            "startup_import": round(startup_seconds, 3),
            "rag_refresh": round(refresh_seconds, 3),
            "offline_token_count": round(token_seconds, 3),
        },
        "offline_token_count": counted_tokens,
    }


def _write_baseline(results: dict) -> None:
    baseline = {
        "format": BASELINE_FORMAT,
        "description": (
            "Ratios measured on the machine below, relative to a fixed "
            "pure-Python calibration workload. Only 'ratios' is compared; "
            "'reference_seconds' is a one-off wall-clock measurement kept for "
            "context and is never used by the check."
        ),
        "reference_measured_at": datetime.date.today().isoformat(),
        "reference_hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "ratios": results["ratios"],
        "reference_seconds": results["reference_seconds"],
    }
    BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_check(update_baseline: bool = False) -> int:
    results = collect_results()
    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["offline_token_count"]:
        print("Offline token counting returned no tokens.")
        return 1

    if update_baseline:
        _write_baseline(results)
        print(f"Wrote baseline to {BASELINE_PATH}.")
        return 0

    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"No usable baseline at {BASELINE_PATH} ({exc}); run --update-baseline.")
        return 1
    if baseline.get("format") != BASELINE_FORMAT:
        print(
            f"Baseline format {baseline.get('format')!r} is not {BASELINE_FORMAT} "
            "and cannot be compared; run --update-baseline."
        )
        return 1
    problems = compare(results["ratios"], baseline.get("ratios", {}))
    if problems:
        for problem in problems:
            print(f"Performance regression: {problem}")
        return 1
    print("Performance check passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record the current measurements as the new baseline",
    )
    arguments = parser.parse_args(argv)
    return run_check(update_baseline=arguments.update_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
