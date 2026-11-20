"""Focused subagent loop used by the run_subagent tool."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..tools import AVAILABLE_TOOLS, approved_tool_invocation

_DEFAULT_SUBAGENT_TOOLS = (
    "read_local_file",
    "search_workspace_text",
    "search_workspace_files",
    "list_directory",
    "inspect_workspace_git",
)


async def run_subagent_loop(
    llm: Any,
    task: str,
    allowed_tools: list[str] | None,
    max_iterations: int,
) -> str:
    """Run a focused agent loop and return only its final summary."""
    subset_names = [name for name in (allowed_tools or []) if name in AVAILABLE_TOOLS]
    if not subset_names:
        subset_names = [
            name for name in _DEFAULT_SUBAGENT_TOOLS if name in AVAILABLE_TOOLS
        ]
    subset_tools = [AVAILABLE_TOOLS[name] for name in subset_names]

    try:
        from .runtime import _bind_tools

        sub_llm_with_tools = _bind_tools(llm, subset_tools) if subset_tools else llm
    except Exception:  # noqa: BLE001 - bind_tools errors depend on the model
        sub_llm_with_tools = llm

    messages = [
        SystemMessage(
            content=(
                "You are a focused subagent. Use the available tools to "
                "investigate the task, then reply with a concise summary."
            )
        ),
        HumanMessage(content=task),
    ]

    for _ in range(max_iterations):
        response = await sub_llm_with_tools.ainvoke(messages)
        if not getattr(response, "tool_calls", None):
            return str(getattr(response, "content", "") or "")
        messages.append(response)
        for tool_call in response.tool_calls:
            name = tool_call.get("name")
            args = tool_call.get("args", {}) or {}
            call_id = tool_call.get("id")
            if name not in subset_names:
                result = f"Error: Tool '{name}' is not allowed for this subagent."
            else:
                tool_func = AVAILABLE_TOOLS.get(name)
                if tool_func is None:
                    result = f"Error: Unknown tool '{name}'."
                else:
                    try:
                        with approved_tool_invocation():
                            if hasattr(tool_func, "ainvoke"):
                                result = await tool_func.ainvoke(args)
                            else:
                                result = await asyncio.to_thread(tool_func.invoke, args)
                    except Exception as exc:  # noqa: BLE001 - tool execution boundary
                        result = f"Error executing {name}: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    return "Subagent reached the iteration limit without a final answer."


def make_subagent_runner(llm):
    async def runner(task, allowed_tools, max_iterations):
        return await run_subagent_loop(llm, task, allowed_tools, max_iterations)

    return runner
