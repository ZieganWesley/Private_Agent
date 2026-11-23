"""Repository intelligence, architecture memory and agent metrics (Part M).

This package implements the high-priority portions of issue #13:
tree-sitter based repository indexing, symbol retrieval, repository maps,
architecture memory, typed memory ranking helpers and agent metrics.
OpenVINO/NPU acceleration is intentionally deferred (future work).
"""

__all__ = [
    "RepositoryIndex",
    "RepositoryGraph",
    "build_repository_graph",
    "get_repository_graph",
    "graph_nodes",
    "graph_edges",
    "graph_neighbors",
    "impact_analysis",
    "trace_path",
    "find_symbol",
    "find_definition",
    "find_references",
    "find_callers",
    "find_callees",
    "find_dependencies",
    "trace_execution_flow",
    "build_repository_map",
    "ArchitectureMemory",
    "AgentMetrics",
    "classify_memory",
    "rank_memories",
]


def __getattr__(name: str):
    if name in {"RepositoryIndex"}:
        from .indexer import RepositoryIndex

        return RepositoryIndex
    if name in {
        "RepositoryGraph",
        "build_repository_graph",
        "get_repository_graph",
        "graph_nodes",
        "graph_edges",
        "graph_neighbors",
        "impact_analysis",
        "trace_path",
    }:
        from . import graph

        return getattr(graph, name)
    if name in {
        "find_symbol",
        "find_definition",
        "find_references",
        "find_callers",
        "find_callees",
        "find_dependencies",
        "trace_execution_flow",
    }:
        from . import queries

        return getattr(queries, name)
    if name == "build_repository_map":
        from .repo_map import build_repository_map

        return build_repository_map
    if name == "ArchitectureMemory":
        from .architecture_memory import ArchitectureMemory

        return ArchitectureMemory
    if name == "AgentMetrics":
        from .metrics import AgentMetrics

        return AgentMetrics
    if name in {"classify_memory", "rank_memories"}:
        from .memory_ranking import classify_memory, rank_memories

        return {"classify_memory": classify_memory, "rank_memories": rank_memories}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
