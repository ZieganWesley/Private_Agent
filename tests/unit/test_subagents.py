import asyncio

from private_agent.sandbox import SandboxManager
from private_agent.tools import get_tool_context, set_tool_context, subagents


def test_run_subagent_requires_runner(tmp_path):
    SandboxManager.set_root(str(tmp_path))
    ctx = get_tool_context()
    ctx.subagent_runner = None
    set_tool_context(ctx)
    result = asyncio.run(subagents.run_subagent.ainvoke({"task": "find files"}))
    assert "unavailable" in result


def test_run_subagent_empty_task(tmp_path):
    SandboxManager.set_root(str(tmp_path))
    ctx = get_tool_context()
    ctx.subagent_runner = None
    set_tool_context(ctx)
    result = asyncio.run(subagents.run_subagent.ainvoke({"task": "   "}))
    assert "must not be empty" in result


def test_run_subagent_calls_runner(tmp_path):
    SandboxManager.set_root(str(tmp_path))

    async def runner(task, allowed_tools, max_iterations):
        assert task == "summarize"
        assert "read_local_file" in allowed_tools
        assert max_iterations == 3
        return "done summary"

    ctx = get_tool_context()
    ctx.subagent_runner = runner
    set_tool_context(ctx)
    result = asyncio.run(
        subagents.run_subagent.ainvoke(
            {
                "task": "summarize",
                "allowed_tools": ["read_local_file"],
                "max_iterations": 3,
            }
        )
    )
    assert result == "done summary"


def test_run_subagent_handles_runner_failure(tmp_path):
    SandboxManager.set_root(str(tmp_path))

    async def runner(task, allowed_tools, max_iterations):
        raise RuntimeError("boom")

    ctx = get_tool_context()
    ctx.subagent_runner = runner
    set_tool_context(ctx)
    result = asyncio.run(subagents.run_subagent.ainvoke({"task": "summarize"}))
    assert "Error running subagent" in result
