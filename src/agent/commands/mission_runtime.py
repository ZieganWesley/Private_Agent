"""Runtime bridge for the /mission slash command."""

from __future__ import annotations


def mission_slash_command(user_input, memory, session_id, session_ui):
    """Handle /mission; return the goal for a new mission or None to continue."""
    if user_input.lower() != "/mission" and not user_input.lower().startswith(
        "/mission "
    ):
        return user_input
    from .mission import handle_mission_command

    result = handle_mission_command(
        memory=memory,
        session_id=session_id,
        argument=user_input[len("/mission") :].strip(),
        session_ui=session_ui,
    )
    if result and result.get("action") == "start":
        controller = result["controller"]
        controller.begin_iteration()
        controller.next_step()
        return controller.mission()["goal"]
    return None


__all__ = ["mission_slash_command"]
