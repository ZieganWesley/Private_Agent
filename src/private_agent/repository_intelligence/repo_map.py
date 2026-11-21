"""Persistent repository maps and structure snapshots."""

from __future__ import annotations

import time
from pathlib import Path

from .indexer import RepositoryIndex


def build_repository_map(root_path: str = ".", db_path: str | None = None) -> str:
    root = Path(root_path).expanduser().resolve()
    index = RepositoryIndex(db_path=db_path, root_path=str(root))
    try:
        index.index_repository(root, incremental=True)
        # gather symbols by file
        cur = index.conn.cursor()
        cur.execute(
            "SELECT f.path, s.name, s.kind, s.start_line FROM files f LEFT JOIN symbols s ON s.file_id=f.id ORDER BY f.path, s.start_line"
        )
        lines = [f"# Repository map: {root}", ""]
        current = None
        for path, name, kind, line in cur.fetchall():
            if path != current:
                current = path
                lines.append(f"## {path}")
            if name:
                lines.append(f"- {kind} {name} (line {line})")
        # store snapshot
        try:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS repo_maps (id INTEGER PRIMARY KEY, root TEXT, version TEXT, content TEXT, created_at REAL)"
            )
            cur.execute(
                "INSERT INTO repo_maps(root, version, content, created_at) VALUES(?,?,?,?)",
                (str(root), "", "\n".join(lines), time.time()),
            )
            index.conn.commit()
        except Exception:  # noqa: BLE001 - repo map storage is best-effort
            pass
        return "\n".join(lines)
    finally:
        index.close()
