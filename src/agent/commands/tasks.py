"""Handler for the /tasks slash command."""


async def handle_tasks_command(
    *,
    memory,
    session_id,
    skills_folder_input,
    loaded_skills,
    session_ui,
    console,
):
    """Run task management and return (approved_task_id, loaded_skills)."""
    from agent.runtime import _run_task_management
    from tools import get_tool_context, set_tool_context
    from skills.loader import load_skills_from_folder
    import os

    approved_task_id = await __import__("asyncio").to_thread(
        _run_task_management,
        memory,
        session_id,
        skills_folder_input or None,
    )
    loaded_skills = (
        load_skills_from_folder(skills_folder_input)
        if skills_folder_input
        else {}
    )
    _tool_context = get_tool_context()
    _tool_context.skill_directory = (
        os.path.abspath(os.path.expanduser(skills_folder_input))
        if skills_folder_input
        else None
    )
    _tool_context.skill_registry = loaded_skills
    set_tool_context(_tool_context)

    if approved_task_id:
        approved_task = memory.get_task(approved_task_id)
        if not (
            approved_task
            and approved_task["status"] == "active"
            and approved_task["approved_revision"] == approved_task["revision"]
            and approved_task["approved_digest"] == approved_task["digest"]
        ):
            session_ui.print(
                "[warning][Task not started][/warning] The task is no "
                "longer active with current approval; no execution "
                "was started. Review it with `/tasks`."
            )
            return None, loaded_skills
        user_input = approved_task["goal"]
        session_ui.print(
            "[success][Task started][/success] Continuing with the "
            "approved plan automatically; no need to repeat the "
            "request. Enter `pause` via `/tasks` to stop it."
        )
        return approved_task_id, loaded_skills, user_input
    return None, loaded_skills, None
