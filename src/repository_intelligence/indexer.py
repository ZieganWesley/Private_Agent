"""Tree-sitter based repository indexer.

Stores files, symbols and relationships in a project-scoped SQLite database.
Supports incremental re-indexing by file hash and repository version.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import guard exercised in tests via installed package
    from tree_sitter import Language, Parser
except Exception:  # noqa: BLE001 - optional dependency
    Language = None  # type: ignore[assignment]
    Parser = None  # type: ignore[assignment]

_LOGGER = logging.getLogger("private_agent.repository_intelligence.indexer")

_LANGUAGE_MODULES = {
    ".py": ("tree_sitter_python", "python"),
    ".js": ("tree_sitter_javascript", "javascript"),
    ".jsx": ("tree_sitter_javascript", "javascript"),
    ".ts": ("tree_sitter_typescript", "typescript"),
    ".tsx": ("tree_sitter_typescript", "tsx"),
    ".go": ("tree_sitter_go", "go"),
    ".rs": ("tree_sitter_rust", "rust"),
    ".java": ("tree_sitter_java", "java"),
}

_SYMBOL_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "struct_item",
    "enum_item",
    "trait_item",
    "impl_item",
}

_CALL_NODE_TYPES = {"call_expression", "call"}


def _load_language(suffix: str):
    module_name, _ = _LANGUAGE_MODULES[suffix]
    try:
        module = __import__(module_name)
    except Exception:  # noqa: BLE001 - tree-sitter language factory may raise
        return None
    try:
        if suffix == ".tsx":
            factory = getattr(module, "language_tsx", None) or getattr(module, "language", None)
        elif suffix == ".ts":
            factory = getattr(module, "language_typescript", None) or getattr(module, "language", None)
        else:
            factory = getattr(module, "language", None)
        if factory is None:
            return None
        return Language(factory())
    except Exception:  # noqa: BLE001 - tree-sitter language factory may raise
        return None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _git_version(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git may be unavailable
        pass
    return ""


@dataclass
class Symbol:
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    parent: str = ""


class RepositoryIndex:
    """SQLite-backed repository symbol index."""

    def __init__(self, db_path: str | None = None, root_path: str | None = None):
        if db_path is None:
            if root_path is not None:
                db_path = str(Path(root_path).expanduser().resolve() / ".private_agent" / "repo_index.db")
            else:
                db_path = str(Path.cwd() / ".private_agent" / "repo_index.db")
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.root_path = str(Path(root_path).expanduser().resolve()) if root_path else None
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY,
                root TEXT UNIQUE,
                version TEXT,
                indexed_at REAL
            );
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                repository_id INTEGER,
                path TEXT,
                language TEXT,
                hash TEXT,
                mtime REAL,
                indexed_at REAL,
                UNIQUE(repository_id, path)
            );
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                name TEXT,
                kind TEXT,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                parent TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY,
                file_id INTEGER,
                source_symbol TEXT,
                target_name TEXT,
                kind TEXT
            );
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY,
                repository_id INTEGER,
                kind TEXT,
                name TEXT,
                file_path TEXT,
                UNIQUE(repository_id, kind, name, file_path)
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY,
                repository_id INTEGER,
                source TEXT,
                target TEXT,
                kind TEXT,
                file_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON graph_nodes(name);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_symbol);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_name);
            """
        )
        self.conn.commit()

    def graph_nodes(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Return repository knowledge-graph nodes."""
        with self._lock:
            cur = self.conn.cursor()
            if kind:
                cur.execute(
                    "SELECT id, kind, name, file_path FROM graph_nodes WHERE kind=?",
                    (kind,),
                )
            else:
                cur.execute("SELECT id, kind, name, file_path FROM graph_nodes")
            return [
                {"id": row[0], "kind": row[1], "name": row[2], "file": row[3]}
                for row in cur.fetchall()
            ]

    def graph_edges(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Return repository knowledge-graph edges."""
        with self._lock:
            cur = self.conn.cursor()
            if kind:
                cur.execute(
                    "SELECT id, source, target, kind, file_path FROM graph_edges WHERE kind=?",
                    (kind,),
                )
            else:
                cur.execute("SELECT id, source, target, kind, file_path FROM graph_edges")
            return [
                {"id": row[0], "source": row[1], "target": row[2], "kind": row[3], "file": row[4]}
                for row in cur.fetchall()
            ]

    def graph_neighbors(self, node: str, direction: str = "both") -> list[dict[str, Any]]:
        """Return edges incident to a graph node."""
        with self._lock:
            cur = self.conn.cursor()
            rows = []
            if direction in {"out", "both"}:
                cur.execute(
                    "SELECT source, target, kind, file_path FROM graph_edges WHERE source=?",
                    (node,),
                )
                rows.extend(
                    {"source": r[0], "target": r[1], "kind": r[2], "file": r[3], "direction": "out"}
                    for r in cur.fetchall()
                )
            if direction in {"in", "both"}:
                cur.execute(
                    "SELECT source, target, kind, file_path FROM graph_edges WHERE target=?",
                    (node,),
                )
                rows.extend(
                    {"source": r[0], "target": r[1], "kind": r[2], "file": r[3], "direction": "in"}
                    for r in cur.fetchall()
                )
            return rows

    def impact_analysis(self, target: str, max_depth: int = 5) -> list[dict[str, Any]]:
        """Return nodes that can reach or be reached by *target*."""
        seen: set[str] = {target}
        frontier = [target]
        impacted: list[dict[str, Any]] = []
        depth = 0
        while frontier and depth < max_depth:
            next_frontier: list[str] = []
            with self._lock:
                cur = self.conn.cursor()
                for node in frontier:
                    cur.execute(
                        "SELECT source, target, kind, file_path FROM graph_edges WHERE source=? OR target=?",
                        (node, node),
                    )
                    for source, target, kind, file_path in cur.fetchall():
                        other = target if source == node else source
                        if other not in seen:
                            seen.add(other)
                            next_frontier.append(other)
                            impacted.append(
                                {
                                    "node": other,
                                    "via": node,
                                    "kind": kind,
                                    "file": file_path,
                                    "depth": depth + 1,
                                }
                            )
            frontier = next_frontier
            depth += 1
        return impacted

    def path_between(self, source: str, target: str, max_depth: int = 8) -> list[str] | None:
        """Find a shortest path between two graph nodes."""
        if source == target:
            return [source]
        visited = {source}
        queue: list[tuple[str, list[str]]] = [(source, [source])]
        depth = 0
        while queue and depth < max_depth:
            next_queue: list[tuple[str, list[str]]] = []
            for node, path in queue:
                with self._lock:
                    cur = self.conn.cursor()
                    cur.execute(
                        "SELECT target FROM graph_edges WHERE source=? UNION SELECT source FROM graph_edges WHERE target=?",
                        (node, node),
                    )
                    neighbors = [r[0] for r in cur.fetchall()]
                for neighbor in neighbors:
                    if neighbor == target:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_queue.append((neighbor, path + [neighbor]))
            queue = next_queue
            depth += 1
        return None

    def to_mermaid(self) -> str:
        """Render the repository graph as Mermaid."""
        lines = ["graph TD"]
        for node in self.graph_nodes():
            safe = node["name"].replace("\"", "'")
            lines.append(f'  N{node["id"]}["{safe}"]')
        for edge in self.graph_edges():
            # Mermaid needs node identifiers; use names as stable labels.
            lines.append(f'  "{edge["source"]}" -->|{edge["kind"]}| "{edge["target"]}"')
        return "\n".join(lines)

    def close(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):
            self.conn.close()

    def _repository_id(self, root: Path) -> int:
        version = _git_version(root)
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO repositories(root, version, indexed_at) VALUES(?,?,?) "
                "ON CONFLICT(root) DO UPDATE SET version=excluded.version, indexed_at=excluded.indexed_at",
                (str(root), version, time.time()),
            )
            self.conn.commit()
            cur.execute("SELECT id FROM repositories WHERE root=?", (str(root),))
            return int(cur.fetchone()[0])

    def index_repository(self, root: str | Path = ".", incremental: bool = True) -> dict[str, Any]:
        root = Path(root).expanduser().resolve()
        repo_id = self._repository_id(root)
        files_indexed = 0
        symbols_indexed = 0
        files_skipped = 0
        pending: list[tuple[Path, str, str, int]] = []
        for path in self._iter_files(root):
            suffix = path.suffix.lower()
            if suffix not in _LANGUAGE_MODULES:
                continue
            rel = str(path.relative_to(root))
            file_hash = _file_hash(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if incremental:
                cur = self.conn.cursor()
                cur.execute(
                    "SELECT id, hash FROM files WHERE repository_id=? AND path=?",
                    (repo_id, rel),
                )
                row = cur.fetchone()
                if row and row[1] == file_hash:
                    files_skipped += 1
                    continue
                if row:
                    file_id = row[0]
                    cur.execute("DELETE FROM symbols WHERE file_id=?", (file_id,))
                    cur.execute("DELETE FROM edges WHERE file_id=?", (file_id,))
                    cur.execute(
                        "UPDATE files SET language=?, hash=?, mtime=?, indexed_at=? WHERE id=?",
                        (_LANGUAGE_MODULES[suffix][1], file_hash, mtime, time.time(), file_id),
                    )
                else:
                    cur.execute(
                        "INSERT INTO files(repository_id, path, language, hash, mtime, indexed_at) VALUES(?,?,?,?,?,?)",
                        (repo_id, rel, _LANGUAGE_MODULES[suffix][1], file_hash, mtime, time.time()),
                    )
                    file_id = cur.lastrowid
                self.conn.commit()
            else:
                with self._lock:
                    cur = self.conn.cursor()
                    cur.execute("SELECT id FROM files WHERE repository_id=? AND path=?", (repo_id, rel))
                    old_row = cur.fetchone()
                    if old_row:
                        cur.execute("DELETE FROM symbols WHERE file_id=?", (old_row[0],))
                        cur.execute("DELETE FROM edges WHERE file_id=?", (old_row[0],))
                    cur.execute(
                        "DELETE FROM files WHERE repository_id=? AND path=?",
                        (repo_id, rel),
                    )
                    cur.execute(
                        "INSERT INTO files(repository_id, path, language, hash, mtime, indexed_at) VALUES(?,?,?,?,?,?)",
                        (repo_id, rel, _LANGUAGE_MODULES[suffix][1], file_hash, mtime, time.time()),
                    )
                    file_id = cur.lastrowid
                    self.conn.commit()
            pending.append((path, rel, suffix, int(file_id)))
        parsed = self._parse_files_parallel(
            [(path, rel, suffix) for path, rel, suffix, _ in pending]
        )
        for (_path, _rel, _suffix, file_id), (syms, edges) in zip(pending, parsed, strict=False):
            with self._lock:
                cur = self.conn.cursor()
                for sym in syms:
                    cur.execute(
                        "INSERT INTO symbols(file_id, name, kind, start_line, end_line, signature, parent) VALUES(?,?,?,?,?,?,?)",
                        (file_id, sym.name, sym.kind, sym.start_line, sym.end_line, sym.signature, sym.parent),
                    )
                    symbols_indexed += 1
                for edge in edges:
                    cur.execute(
                        "INSERT INTO edges(file_id, source_symbol, target_name, kind) VALUES(?,?,?,?)",
                        (file_id, edge[0], edge[1], edge[2]),
                    )
                self.conn.commit()
            files_indexed += 1
        # remove deleted files
        self._prune_deleted(root, repo_id)
        self._rebuild_graph(repo_id)
        return {
            "root": str(root),
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "symbols_indexed": symbols_indexed,
            "version": _git_version(root),
        }

    def _iter_files(self, root: Path) -> Iterable[Path]:
        ignore_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]
            for name in filenames:
                yield Path(dirpath) / name

    def _prune_deleted(self, root: Path, repo_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT id, path FROM files WHERE repository_id=?", (repo_id,))
        rows = cur.fetchall()
        for fid, rel in rows:
            if not (root / rel).exists():
                with self._lock:
                    c = self.conn.cursor()
                    c.execute("DELETE FROM symbols WHERE file_id=?", (fid,))
                    c.execute("DELETE FROM edges WHERE file_id=?", (fid,))
                    c.execute("DELETE FROM files WHERE id=?", (fid,))
                    self.conn.commit()

    def _rebuild_graph(self, repo_id: int) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM graph_nodes WHERE repository_id=?", (repo_id,))
            cur.execute("DELETE FROM graph_edges WHERE repository_id=?", (repo_id,))
            cur.execute("SELECT id, path FROM files WHERE repository_id=?", (repo_id,))
            files = cur.fetchall()
            for _fid, path in files:
                cur.execute(
                    "INSERT OR IGNORE INTO graph_nodes(repository_id, kind, name, file_path) VALUES(?,?,?,?)",
                    (repo_id, "file", path, path),
                )
            cur.execute(
                "SELECT s.id, s.name, s.kind, f.path FROM symbols s JOIN files f ON s.file_id=f.id WHERE f.repository_id=?",
                (repo_id,),
            )
            for _sid, name, kind, path in cur.fetchall():
                cur.execute(
                    "INSERT OR IGNORE INTO graph_nodes(repository_id, kind, name, file_path) VALUES(?,?,?,?)",
                    (repo_id, kind or "symbol", name, path),
                )
                cur.execute(
                    "INSERT INTO graph_edges(repository_id, source, target, kind, file_path) VALUES(?,?,?,?,?)",
                    (repo_id, path, name, "contains", path),
                )
            cur.execute(
                "SELECT e.source_symbol, e.target_name, e.kind, f.path FROM edges e JOIN files f ON e.file_id=f.id WHERE f.repository_id=?",
                (repo_id,),
            )
            for source, target, kind, path in cur.fetchall():
                if source and target:
                    cur.execute(
                        "INSERT INTO graph_edges(repository_id, source, target, kind, file_path) VALUES(?,?,?,?,?)",
                        (repo_id, source, target, kind or "reference", path),
                    )
            self.conn.commit()

    def _parse_file(self, path: Path, rel: str, suffix: str) -> tuple[list[Symbol], list[tuple[str, str, str]]]:
        return _parse_file_impl(path, rel, suffix)

    def _parse_files_parallel(
        self, pending: list[tuple[Path, str, str]]
    ) -> list[tuple[list[Symbol], list[tuple[str, str, str]]]]:
        if not pending:
            return []
        cpu_count = os.cpu_count() or 1
        if len(pending) < 8 or cpu_count < 2:
            return [_parse_file_impl(path, rel, suffix) for path, rel, suffix in pending]
        payloads = [(str(path), rel, suffix) for path, rel, suffix in pending]
        chunk_size = max(1, len(payloads) // (cpu_count * 4))
        try:
            with ProcessPoolExecutor(max_workers=min(len(payloads), cpu_count)) as executor:
                results = list(executor.map(_parse_worker, payloads, chunksize=chunk_size))
            return results
        except Exception:  # noqa: BLE001 - fall back to in-process parsing
            return [_parse_file_impl(path, rel, suffix) for path, rel, suffix in pending]

    # Query helpers
    def find_symbols(self, name: str, kind: str | None = None) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        if kind:
            cur.execute(
                "SELECT s.id, s.name, s.kind, f.path, s.start_line, s.end_line, s.parent FROM symbols s JOIN files f ON s.file_id=f.id WHERE s.name=? AND s.kind=?",
                (name, kind),
            )
        else:
            cur.execute(
                "SELECT s.id, s.name, s.kind, f.path, s.start_line, s.end_line, s.parent FROM symbols s JOIN files f ON s.file_id=f.id WHERE s.name=?",
                (name,),
            )
        return [
            {"id": r[0], "name": r[1], "kind": r[2], "file": r[3], "start_line": r[4], "end_line": r[5], "parent": r[6]}
            for r in cur.fetchall()
        ]

    def find_references(self, name: str) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT e.source_symbol, f.path, e.kind FROM edges e JOIN files f ON e.file_id=f.id WHERE e.target_name = ?",
            (name,),
        )
        return [{"source": r[0], "file": r[1], "kind": r[2]} for r in cur.fetchall()]

    def callers_of(self, name: str) -> list[dict[str, Any]]:
        refs = self.find_references(name)
        return [r for r in refs if r["kind"] == "calls"]

    def callees_of(self, name: str) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT target_name, f.path FROM edges e JOIN files f ON e.file_id=f.id WHERE source_symbol=? AND kind='calls'",
            (name,),
        )
        return [{"target": r[0], "file": r[1]} for r in cur.fetchall()]

    def dependencies_of(self, path_or_symbol: str) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT target_name, f.path, e.kind FROM edges e JOIN files f ON e.file_id=f.id WHERE (f.path=? OR e.source_symbol=?) AND e.kind='imports'",
            (path_or_symbol, path_or_symbol),
        )
        return [{"target": r[0], "file": r[1], "kind": r[2]} for r in cur.fetchall()]

    def trace_flow(self, entry: str, depth: int = 3) -> list[dict[str, Any]]:
        result = []
        seen = set()
        frontier = [(entry, 0)]
        while frontier:
            sym, d = frontier.pop()
            if d > depth or sym in seen:
                continue
            seen.add(sym)
            result.append({"symbol": sym, "depth": d})
            for c in self.callees_of(sym):
                frontier.append((c["target"], d + 1))
        return result

def _extract_identifiers(text: str) -> list[str]:
    import re

    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)

def _import_target(node_type: str, text: str) -> str:
    import re

    text = text.strip()
    # Python: from x import y / import x
    m = re.search(r"^from\s+([.\w]+)\s+import", text)
    if m:
        return m.group(1)
    m = re.search(r"^import\s+([.\w]+)", text)
    if m:
        return m.group(1)
    # JS/TS: ... from 'mod' or require('mod')
    m = re.search(r"from\s+['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1)
    m = re.search(r"require\(['\"]([^'\"]+)['\"]\)", text)
    if m:
        return m.group(1)
    # Go/Java style: import "mod"
    m = re.search(r"import\s+['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1)
    return ""

def _parse_file_impl(path: Path, rel: str, suffix: str) -> tuple[list[Symbol], list[tuple[str, str, str]]]:
    language = _load_language(suffix)
    if language is None or Parser is None:
        return [], []
    try:
        source = path.read_bytes()
    except OSError:
        return [], []
    parser = Parser(language)
    tree = parser.parse(source)
    symbols: list[Symbol] = []
    edges: list[tuple[str, str, str]] = []
    _walk(tree.root_node, source, "", symbols, edges, rel)
    return symbols, edges

def _walk(node, source: bytes, parent: str, symbols: list[Symbol], edges: list[tuple[str, str, str]], rel: str) -> None:
    ntype = node.type
    name = ""
    kind = ""
    current_parent = parent
    if ntype in _SYMBOL_NODE_TYPES:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
        kind = ntype.replace("_declaration", "").replace("_definition", "").replace("_item", "")
        if name:
            signature = ""
            for child in node.children:
                if "parameters" in child.type or "parameter" in child.type:
                    signature = source[child.start_byte:child.end_byte].decode("utf-8", "replace")
                    break
            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    file_path=rel,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    signature=signature,
                    parent=parent,
                )
            )
            # inheritance edges for classes
            if "class" in kind:
                for child in node.children:
                    if "heritage" in child.type or "superclass" in child.type or "bases" in child.type or "argument_list" in child.type:
                        text = source[child.start_byte:child.end_byte].decode("utf-8", "replace")
                        for base in _extract_identifiers(text):
                            if base and base != name:
                                edges.append((name, base, "inherits"))
            current_parent = name or parent
    # calls
    if ntype in _CALL_NODE_TYPES:
        func_node = node.child_by_field_name("function") or node.child_by_field_name("callee")
        if func_node is not None:
            callee = source[func_node.start_byte:func_node.end_byte].decode("utf-8", "replace")
            callee = callee.split(".")[-1].split("(")[0].strip()
            if callee and callee.isidentifier():
                edges.append((current_parent or "<module>", callee, "calls"))
    # imports
    if "import" in ntype:
        text = source[node.start_byte:node.end_byte].decode("utf-8", "replace")
        target = _import_target(ntype, text)
        if target:
            edges.append((current_parent or "<module>", target, "imports"))
    for child in node.children:
        _walk(child, source, current_parent, symbols, edges, rel)



def _parse_worker(payload: tuple[str, str, str]) -> tuple[list[Symbol], list[tuple[str, str, str]]]:
    path_str, rel, suffix = payload
    return _parse_file_impl(Path(path_str), rel, suffix)


if __name__ == "__main__":
    pass
