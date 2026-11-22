"""Symbol retrieval helpers for the repository index."""

from __future__ import annotations

from .indexer import RepositoryIndex


def _index(root_path: str = ".", db_path: str | None = None) -> RepositoryIndex:
    return RepositoryIndex(db_path=db_path, root_path=root_path)


def find_symbol(name: str, root_path: str = ".", db_path: str | None = None, kind: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.find_symbols(name, kind)
    finally:
        index.close()


def find_definition(name: str, root_path: str = ".", db_path: str | None = None):
    results = find_symbol(name, root_path=root_path, db_path=db_path)
    return results[0] if results else None


def find_references(name: str, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.find_references(name)
    finally:
        index.close()


def find_callers(name: str, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.callers_of(name)
    finally:
        index.close()


def find_callees(name: str, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.callees_of(name)
    finally:
        index.close()


def find_dependencies(target: str, root_path: str = ".", db_path: str | None = None):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.dependencies_of(target)
    finally:
        index.close()


def trace_execution_flow(entry: str, root_path: str = ".", db_path: str | None = None, depth: int = 3):
    index = _index(root_path, db_path)
    try:
        index.index_repository(root_path, incremental=True)
        return index.trace_flow(entry, depth=depth)
    finally:
        index.close()
