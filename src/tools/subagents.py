"""Subagent delegation tool."""

import logging

from langchain_core.tools import tool

from . import get_tool_context
from .schemas import RunSubagentInput

logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_TOOLS = (
    "read_local_file",
    "search_workspace_text",
    "search_workspace_files",
    "list_directory",
    "inspect_workspace_git",
)


@tool(args_schema=RunSubagentInput)
async def run_subagent(
    task: str, allowed_tools: list[str] | None = None, max_iterations: int = 5
) -> str:
    """Run a focused subagent and return only its summary."""
    cleaned_task = task.strip()
    if not cleaned_task:
        return "Error: Task must not be empty."
    runner = get_tool_context().subagent_runner
    if runner is None:
        return "Error: Subagent runner is unavailable in this session."
    tool_names = allowed_tools if allowed_tools else list(DEFAULT_SUBAGENT_TOOLS)
    try:
        result = await runner(cleaned_task, tool_names, max_iterations)
    except Exception as exc:  # noqa: BLE001 - subagent failures should not crash the main turn
        logger.exception("Subagent failed")
        return f"Error running subagent: {exc}"
    if not isinstance(result, str):
        result = str(result)
    return result.strip() or "Subagent completed without a summary."
