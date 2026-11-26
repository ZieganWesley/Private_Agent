"""Handler for the /mission slash command."""

from __future__ import annotations

import shlex

from agent.missions import MissionController


def _print(session_ui, text: str) -> None:
    if hasattr(session_ui, "print"):
        session_ui.print(text)
    elif hasattr(session_ui, "print_markup"):
        session_ui.print_markup(text)


def _mission_line(mission: dict) -> str:
    return (
        f"{mission['mission_id']}  {mission['status']:<12} "
        f"iter {mission['iteration']}/{mission['max_iterations']}  "
        f"{mission['goal'][:80]}"
    )


def handle_mission_command(
    *,
    memory,
    session_id: str,
    argument: str,
    session_ui,
) -> dict | None:
    """Handle /mission subcommands and return the created controller, if any."""
    try:
        parts = shlex.split(argument or "")
    except ValueError as exc:
        _print(session_ui, f"[warning][Mission][/warning] {exc}")
        return None
    if not parts:
        _print(
            session_ui,
            "[info][Mission][/info] Usage: /mission start <goal> | list | "
            "status <id> | pause <id> | resume <id> | stop <id>",
        )
        return None
    action = parts[0].lower()
    if action == "start":
        goal = " ".join(parts[1:]).strip()
        if not goal:
            _print(session_ui, "[warning][Mission][/warning] A goal is required.")
            return None
        controller = MissionController.start(
            memory,
            session_id,
            goal,
            [
                {
                    "step_id": "step-1",
                    "description": f"Plan the work for: {goal}",
                    "acceptance": ["the approach and risks are explicit"],
                },
                {
                    "step_id": "step-2",
                    "description": f"Do the work for: {goal}",
                    "acceptance": ["the requested outcome is implemented"],
                },
                {
                    "step_id": "step-3",
                    "description": f"Verify the work for: {goal}",
                    "acceptance": ["verification evidence is reported"],
                },
            ],
        )
        _print(
            session_ui,
            f"[success][Mission started][/success] {controller.mission_id} — {goal}",
        )
        return {"action": "start", "controller": controller}
    if action == "list":
        missions = memory.list_missions(session_id=session_id)
        if not missions:
            _print(session_ui, "[info][Mission][/info] No missions in this session.")
        for mission in missions:
            _print(session_ui, _mission_line(mission))
        return {"action": "list", "missions": missions}
    if action in {"status", "pause", "resume", "stop"}:
        if len(parts) < 2:
            _print(session_ui, f"[warning][Mission][/warning] {action} needs an ID.")
            return None
        mission = memory.get_mission(parts[1])
        if mission is None:
            _print(session_ui, "[warning][Mission][/warning] Mission not found.")
            return None
        try:
            if action == "pause":
                mission = memory.update_mission(parts[1], status="paused")
            elif action == "resume":
                mission = memory.update_mission(parts[1], status="active")
            elif action == "stop":
                mission = memory.update_mission(parts[1], status="stopped")
        except ValueError as exc:
            _print(session_ui, f"[warning][Mission][/warning] {exc}")
            return None
        _print(session_ui, _mission_line(mission))
        return {"action": action, "mission": mission}
    _print(
        session_ui,
        "[warning][Mission][/warning] Unknown action. Use start, list, status, "
        "pause, resume, or stop.",
    )
    return None


__all__ = ["handle_mission_command"]
