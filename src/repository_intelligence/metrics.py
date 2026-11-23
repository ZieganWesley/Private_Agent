"""Agent metrics and observability helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class AgentMetrics:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path.home() / ".config" / "private_agent" / "metrics.db")
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metric_events (
                id INTEGER PRIMARY KEY,
                kind TEXT,
                name TEXT,
                value REAL,
                labels TEXT,
                created_at REAL
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.conn.close()

    def record(
        self,
        kind: str,
        name: str,
        value: float = 1.0,
        labels: dict[str, Any] | None = None,
    ) -> None:
        import json

        self.conn.execute(
            "INSERT INTO metric_events(kind,name,value,labels,created_at) VALUES(?,?,?,?,?)",
            (kind, name, float(value), json.dumps(labels or {}), time.time()),
        )
        self.conn.commit()

    def summary(self) -> dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT kind, name, COUNT(*), SUM(value) FROM metric_events GROUP BY kind, name"
        )
        rows = cur.fetchall()
        items = [
            {"kind": r[0], "name": r[1], "count": r[2], "total": r[3]} for r in rows
        ]
        return {"metrics": items, "total_events": sum(r[2] for r in rows)}

    def health_report(self) -> str:
        s = self.summary()
        lines = ["# Agent health report", ""]
        for m in s["metrics"]:
            lines.append(
                f"- {m['kind']}/{m['name']}: count={m['count']} total={m['total']}"
            )
        if not s["metrics"]:
            lines.append("No metrics recorded yet.")
        return "\n".join(lines)
