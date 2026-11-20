"""Task timeline with estimated finish and step timings (Part F6)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_step_timings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    pack TEXT NOT NULL,
    estimated_seconds REAL NOT NULL,
    actual_seconds REAL,
    confidence REAL NOT NULL DEFAULT 0.5
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def record_estimate(
    conn: sqlite3.Connection,
    run_id: str,
    step_index: int,
    pack: str,
    estimated: float,
    confidence: float = 0.5,
) -> None:
    conn.execute(
        "INSERT INTO agent_step_timings (run_id, step_index, pack, estimated_seconds, confidence) VALUES (?,?,?,?,?)",
        (run_id, step_index, pack, estimated, confidence),
    )
    conn.commit()


def record_actual(
    conn: sqlite3.Connection, run_id: str, step_index: int, actual: float
) -> None:
    conn.execute(
        "UPDATE agent_step_timings SET actual_seconds=? WHERE run_id=? AND step_index=?",
        (actual, run_id, step_index),
    )
    conn.commit()


def median_for_pack(conn: sqlite3.Connection, pack: str) -> float | None:
    cur = conn.execute(
        "SELECT actual_seconds FROM agent_step_timings WHERE pack=? AND actual_seconds IS NOT NULL",
        (pack,),
    )
    vals = [r[0] for r in cur.fetchall()]
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


@dataclass
class StepEstimate:
    seconds: float
    confidence: float


def estimate_step(conn: sqlite3.Connection, pack: str, fallback: float = 5.0) -> StepEstimate:
    med = median_for_pack(conn, pack)
    if med is not None:
        return StepEstimate(seconds=med, confidence=0.8)
    return StepEstimate(seconds=fallback, confidence=0.3)
