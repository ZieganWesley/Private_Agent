"""Architecture memory: decisions and facts with cross-session recall."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class ArchitectureMemory:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path.home() / ".config" / "private_agent" / "architecture.db")
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS architecture_decisions (
                id INTEGER PRIMARY KEY,
                root TEXT,
                title TEXT,
                decision TEXT,
                tradeoffs TEXT,
                rejected TEXT,
                constraints TEXT,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS architecture_facts (
                id INTEGER PRIMARY KEY,
                root TEXT,
                kind TEXT,
                content TEXT,
                created_at REAL
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):
            self.conn.close()

    def record_decision(self, root: str = ".", title: str = "", decision: str = "", tradeoffs: str = "", rejected: str = "", constraints: str = "") -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO architecture_decisions(root,title,decision,tradeoffs,rejected,constraints,created_at) VALUES(?,?,?,?,?,?,?)",
            (root, title, decision, tradeoffs, rejected, constraints, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_fact(self, root: str = ".", kind: str = "architecture", content: str = "") -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO architecture_facts(root,kind,content,created_at) VALUES(?,?,?,?)",
            (root, kind, content, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recall(self, root: str = ".", query: str = "") -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        results: list[dict[str, Any]] = []
        q = f"%{query}%"
        cur.execute(
            "SELECT id,title,decision,tradeoffs,rejected,constraints,created_at FROM architecture_decisions WHERE root=? AND (title LIKE ? OR decision LIKE ?)",
            (root, q, q),
        )
        for r in cur.fetchall():
            results.append({"type": "decision", "id": r[0], "title": r[1], "decision": r[2], "tradeoffs": r[3], "rejected": r[4], "constraints": r[5], "created_at": r[6]})
        cur.execute(
            "SELECT id,kind,content,created_at FROM architecture_facts WHERE root=? AND content LIKE ?",
            (root, q),
        )
        for r in cur.fetchall():
            results.append({"type": "fact", "id": r[0], "kind": r[1], "content": r[2], "created_at": r[3]})
        return results
