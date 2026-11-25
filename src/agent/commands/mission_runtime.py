"""Runtime bridge for the /mission slash command."""

from __future__ import annotations

_ACTIVE_RUNNER = None


def mission_slash_command(user_input, memory, session_id, session_ui):
    """Handle /mission; return the next automatic mission prompt or None."""
    global _ACTIVE_RUNNER
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
        from agent.mission_runner import MissionAutoRunner
        from agent.missions import MissionLeaseError

        try:
            _ACTIVE_RUNNER = MissionAutoRunner(result["controller"], session_id)
        except MissionLeaseError as exc:
            session_ui.print(f"[warning][Mission][/warning] {exc}")
            return None
        return _ACTIVE_RUNNER.next_prompt()
    if result and result.get("action") in {"stop", "pause"}:
        if _ACTIVE_RUNNER is not None:
            _ACTIVE_RUNNER.controller.release_lease(_ACTIVE_RUNNER.owner)
        _ACTIVE_RUNNER = None
    return None


def mission_after_turn(text: str, *, had_tool_calls: bool, interrupted: bool = False):
    """Advance the active mission after one model turn, or return None."""
    global _ACTIVE_RUNNER
    if _ACTIVE_RUNNER is None:
        return None
    prompt = _ACTIVE_RUNNER.observe(
        text, had_tool_calls=had_tool_calls, interrupted=interrupted
    )
    if prompt is None:
        _ACTIVE_RUNNER = None
    return prompt


def after_turn(text: str, response) -> str | None:
    """Advance the mission using final model output, not tool-call rounds."""
    return mission_after_turn(
        text,
        had_tool_calls=bool(getattr(response, "tool_calls", None)),
        interrupted=bool(
            getattr(response, "additional_kwargs", {}).get("interrupted")
        ),
    )


__all__ = ["after_turn", "mission_after_turn", "mission_slash_command"]
