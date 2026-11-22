"""LangChain tools for repository intelligence and architecture memory."""

from __future__ import annotations

from langchain_core.tools import tool

from repository_intelligence import (
    AgentMetrics,
    ArchitectureMemory,
    build_repository_map,
    find_callees,
    find_callers,
    find_definition,
    find_dependencies,
    find_references,
    find_symbol,
    trace_execution_flow,
)


@tool
def index_repository(root_path: str = ".") -> str:
    """Index a repository into the symbol and dependency store."""
    from repository_intelligence.indexer import RepositoryIndex

    index = RepositoryIndex(root_path=root_path)
    try:
        stats = index.index_repository(root_path, incremental=True)
        return (
            f"Indexed {stats['files_indexed']} files "
            f"({stats['files_skipped']} unchanged), "
            f"{stats['symbols_indexed']} symbols."
        )
    finally:
        index.close()


@tool
def find_symbol_tool(name: str, root_path: str = ".") -> str:
    """Find symbols by name in the repository index."""
    results = find_symbol(name, root_path=root_path)
    if not results:
        return f"No symbol named {name!r} found."
    lines = [f"{r['kind']} {r['name']} in {r['file']}:{r['start_line']}" for r in results]
    return "\n".join(lines)


@tool
def find_definition_tool(name: str, root_path: str = ".") -> str:
    """Find the definition of a symbol."""
    result = find_definition(name, root_path=root_path)
    if not result:
        return f"No definition for {name!r} found."
    return f"{result['kind']} {result['name']} in {result['file']}:{result['start_line']}"


@tool
def find_references_tool(name: str, root_path: str = ".") -> str:
    """Find references to a symbol."""
    results = find_references(name, root_path=root_path)
    if not results:
        return f"No references to {name!r} found."
    return "\n".join(f"{r['kind']} in {r['file']} (from {r['source']})" for r in results)


@tool
def find_callers_tool(name: str, root_path: str = ".") -> str:
    """Find callers of a function or method."""
    results = find_callers(name, root_path=root_path)
    if not results:
        return f"No callers of {name!r} found."
    return "\n".join(f"{r['source']} in {r['file']}" for r in results)


@tool
def find_callees_tool(name: str, root_path: str = ".") -> str:
    """Find callees of a function or method."""
    results = find_callees(name, root_path=root_path)
    if not results:
        return f"No callees for {name!r} found."
    return "\n".join(f"{r['target']} in {r['file']}" for r in results)


@tool
def find_dependencies_tool(target: str, root_path: str = ".") -> str:
    """Find imports/dependencies for a file or symbol."""
    results = find_dependencies(target, root_path=root_path)
    if not results:
        return f"No dependencies for {target!r} found."
    return "\n".join(f"{r['target']} ({r['kind']}) in {r['file']}" for r in results)


@tool
def trace_execution_flow_tool(entry: str, root_path: str = ".") -> str:
    """Trace execution flow from an entry symbol."""
    results = trace_execution_flow(entry, root_path=root_path)
    if not results:
        return f"No flow from {entry!r} found."
    return "\n".join(f"{'  ' * r['depth']}{r['symbol']}" for r in results)


@tool
def get_repository_map(root_path: str = ".") -> str:
    """Build a persistent repository map."""
    return build_repository_map(root_path=root_path)


@tool
def record_architecture_decision(
    title: str,
    decision: str,
    root_path: str = ".",
    tradeoffs: str = "",
    rejected: str = "",
    constraints: str = "",
) -> str:
    """Record an architectural decision."""
    mem = ArchitectureMemory()
    try:
        rid = mem.record_decision(
            root=root_path,
            title=title,
            decision=decision,
            tradeoffs=tradeoffs,
            rejected=rejected,
            constraints=constraints,
        )
        return f"Recorded architecture decision #{rid}."
    finally:
        mem.close()


@tool
def recall_architecture(query: str = "", root_path: str = ".") -> str:
    """Recall architecture decisions and facts."""
    mem = ArchitectureMemory()
    try:
        results = mem.recall(root=root_path, query=query)
        if not results:
            return "No architecture memory found."
        lines = []
        for r in results:
            if r["type"] == "decision":
                lines.append(f"Decision: {r['title']} - {r['decision']}")
            else:
                lines.append(f"Fact ({r['kind']}): {r['content']}")
        return "\n".join(lines)
    finally:
        mem.close()


@tool
def get_agent_metrics() -> str:
    """Return an agent health and metrics report."""
    metrics = AgentMetrics()
    try:
        return metrics.health_report()
    finally:
        metrics.close()
