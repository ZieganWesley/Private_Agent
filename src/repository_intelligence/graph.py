"""Repository knowledge-graph helpers."""

from __future__ import annotations

from .indexer import RepositoryIndex


def _index(root_path: str = ".", db_path: str | None = None) -> RepositoryIndex:
    return RepositoryIndex(db_path=db_path, root_path=root_path)


def build_repository_graph(root_path: str = ".", db_path: str | None = None) -> dict:
    """Index a repository and return its graph summary."""
    index = _index(root_path, db_path)
    try:
        stats = index.index_repository(root_path, incremental=True)
        return {
            "stats": stats,
            "nodes": index.graph_nodes(),
            "edges": index.graph_edges(),
            "mermaid": index.to_mermaid(),
        }
    finally:
        index.close()


def graph_nodes(kind: str | None = None, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.graph_nodes(kind)
    finally:
        index.close()


def graph_edges(kind: str | None = None, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.graph_edges(kind)
    finally:
        index.close()


def graph_neighbors(node: str, direction: str = "both", root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.graph_neighbors(node, direction=direction)
    finally:
        index.close()


def impact_analysis(target: str, max_depth: int = 5, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.impact_analysis(target, max_depth=max_depth)
    finally:
        index.close()


def trace_path(source: str, target: str, max_depth: int = 8, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.path_between(source, target, max_depth=max_depth)
    finally:
        index.close()


def get_repository_graph(root_path: str = ".", db_path: str | None = None) -> str:
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.to_mermaid()
    finally:
        index.close()


class RepositoryGraph:
    """Convenience wrapper around :class:`RepositoryIndex` graph methods."""

    def __init__(self, root_path: str = ".", db_path: str | None = None):
        self.root_path = root_path
        self.db_path = db_path
        self.index = RepositoryIndex(db_path=db_path, root_path=root_path)

    def build(self, incremental: bool = True):
        return self.index.index_repository(self.root_path, incremental=incremental)

    def nodes(self, kind: str | None = None):
        return self.index.graph_nodes(kind)

    def edges(self, kind: str | None = None):
        return self.index.graph_edges(kind)

    def neighbors(self, node: str, direction: str = "both"):
        return self.index.graph_neighbors(node, direction=direction)

    def impact(self, target: str, max_depth: int = 5):
        return self.index.impact_analysis(target, max_depth=max_depth)

    def path(self, source: str, target: str, max_depth: int = 8):
        return self.index.path_between(source, target, max_depth=max_depth)

    def mermaid(self) -> str:
        return self.index.to_mermaid()

    def close(self) -> None:
        self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
