"""Repository intelligence, architecture memory and agent metrics (Part M).

This package implements the high-priority portions of issue #13:
tree-sitter based repository indexing, symbol retrieval, repository maps,
architecture memory, typed memory ranking helpers and agent metrics.
OpenVINO/NPU acceleration is intentionally deferred (future work).
"""

from .architecture_memory import ArchitectureMemory
from .indexer import RepositoryIndex
from .memory_ranking import classify_memory, rank_memories
from .metrics import AgentMetrics
from .queries import (
    find_callees,
    find_callers,
    find_definition,
    find_dependencies,
    find_references,
    find_symbol,
    trace_execution_flow,
)
from .repo_map import build_repository_map

__all__ = [
    "RepositoryIndex",
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
