"""Text-plan fallback for models without tool support."""

import json
import re

from rich.markdown import Markdown

from ..tools import get_tool_context, set_tool_context
from .task_commands import _task_plan_markdown


def create_text_plan(memory, session_id, user_input, llm, console) -> None:
    """Parse a JSON text plan from a non-tool model and persist it."""
    fallback_prompt = (
        "Return only a JSON object for a task plan with keys goal, task_type, "
        "steps, assumptions, and constraints. Each step must have step_id and "
        "description.\n\nUser request:\n" + user_input
    )
    fallback_response = llm.invoke(fallback_prompt)
    fallback_text = getattr(fallback_response, "content", str(fallback_response))
    match = re.search(r"\{.*\}", fallback_text, re.DOTALL)
    payload = json.loads(match.group(0)) if match else {}
    goal = str(payload.get("goal") or user_input)[:2000]
    task_type = str(payload.get("task_type") or "other")
    if task_type not in {"research", "coding", "other"}:
        task_type = "other"
    raw_steps = payload.get("steps") or [{"step_id": "plan", "description": user_input}]
    steps = []
    for index, item in enumerate(raw_steps[:20], 1):
        if not isinstance(item, dict):
            item = {"description": str(item)}
        steps.append(
            {
                "step_id": str(item.get("step_id") or f"step_{index}")[:64],
                "description": str(item.get("description") or user_input)[:1000],
                "dependencies": list(item.get("dependencies") or []),
                "validation": list(item.get("validation") or []),
                "proof": list(item.get("proof") or []),
                "risks": list(item.get("risks") or []),
                "edge_cases": list(item.get("edge_cases") or []),
            }
        )
    if not steps:
        steps = [{"step_id": "plan", "description": user_input[:1000]}]
    task = memory.create_task_plan(
        session_id,
        goal,
        task_type,
        steps,
        assumptions=list(payload.get("assumptions") or []),
        constraints=list(payload.get("constraints") or []),
    )
    _tool_context = get_tool_context()
    _tool_context.task_id = task["task_id"]
    set_tool_context(_tool_context)
    console.print(Markdown(_task_plan_markdown(task)))
    console.print(
        "[green]Text plan saved in SQLite; review and approve it with /tasks.[/green]"
    )
PY