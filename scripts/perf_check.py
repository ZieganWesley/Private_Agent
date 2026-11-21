"""Measure startup, RAG refresh and offline token counting (J13.10).

The check compares the current run with ``scripts/perf_baseline.json`` and
fails when startup import time or the RAG refresh on a fixed corpus grows by
more than 20%. Refresh the baseline deliberately with ``--update-baseline``
after an intentional change; never do it to silence a regression.

Usage:
    python scripts/perf_check.py            # compare with the baseline
    python scripts/perf_check.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_PATH = pathlib.Path(__file__).with_name("perf_baseline.json")
GROWTH_LIMIT = 1.20
CORPUS_FILES = 300
CORPUS_SECTIONS = 8

_STARTUP_CODE = "import private_agent.agent.runtime"

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


def _child_environment(home: pathlib.Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
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


def compare(current: dict[str, float], baseline: dict[str, float]) -> list[str]:
    """Return human-readable regressions over ``GROWTH_LIMIT``."""
    problems = []
    for key in ("startup_import_seconds", "rag_refresh_seconds"):
        previous = baseline.get(key)
        now = current.get(key)
        if not previous or now is None:
            continue
        if now > previous * GROWTH_LIMIT:
            problems.append(
                f"{key} grew from {previous:.3f}s to {now:.3f}s "
                f"(limit {GROWTH_LIMIT:.0%})"
            )
    return problems


def run_check(update_baseline: bool = False) -> int:
    results = {
        "startup_import_seconds": round(measure_startup_import(), 3),
        "rag_refresh_seconds": round(measure_rag_refresh(), 3),
        "offline_token_count_seconds": round(measure_offline_token_count()[0], 3),
    }
    if results["offline_token_count_seconds"] <= 0:
        print("Offline token counting did not complete.")
        return 1
    print(json.dumps(results, indent=2, sort_keys=True))

    if update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote baseline to {BASELINE_PATH}.")
        return 0

    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"No usable baseline at {BASELINE_PATH} ({exc}); run --update-baseline.")
        return 1
    problems = compare(results, baseline)
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
