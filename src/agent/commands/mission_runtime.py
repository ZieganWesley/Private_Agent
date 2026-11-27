"""Runtime bridge for the /mission slash command and live mission controls."""

from __future__ import annotations

_ACTIVE_RUNNER = None


def _print(session_ui, text: str) -> None:
    if session_ui is None:
        return
    if hasattr(session_ui, "print"):
        session_ui.print(text)
    elif hasattr(session_ui, "print_markup"):
        session_ui.print_markup(text)


def mission_controls_hint() -> str:
    """Return the controls available while a mission is running."""
    return "/pause · /resume · /steer <text> · /stop"


def _control_notice(session_ui, text: str) -> None:
    _print(session_ui, f"[info][Mission][/info] {text}")


def control_active_mission(
    command: str,
    argument: str = "",
    *,
    turn_control=None,
    session_ui=None,
) -> bool:
    """Apply a live mission control; return False when no mission is active."""
    global _ACTIVE_RUNNER
    runner = _ACTIVE_RUNNER
    if runner is None:
        return False
    controller = runner.controller
    mission_id = controller.mission_id
    try:
        if command == "pause":
            if controller.status == "paused":
                _control_notice(session_ui, f"Mission {mission_id} is already paused.")
            else:
                controller.pause()
                if turn_control is not None:
                    turn_control.request_stop()
                _print(
                    session_ui,
                    f"[success][Mission paused][/success] {mission_id} · "
                    "/resume to continue · /stop to end",
                )
            return True
        if command == "resume":
            if controller.status != "paused":
                _control_notice(
                    session_ui,
                    f"Mission {mission_id} is {controller.status}; nothing to resume.",
                )
                return True
            controller.resume()
            if turn_control is not None:
                turn_control.clear_stop()
                turn_control.queue_message(
                    "Mission resumed by the user. Continue the current mission step.",
                    mode="interrupt",
                )
            _print(
                session_ui,
                f"[success][Mission resumed][/success] {mission_id} · "
                f"{mission_controls_hint()}",
            )
            return True
        if command == "stop":
            controller.stop()
            controller.release_lease(runner.owner)
            if turn_control is not None:
                turn_control.request_stop()
            _ACTIVE_RUNNER = None
            _print(
                session_ui,
                f"[warning][Mission stopped][/warning] {mission_id}",
            )
            return True
        if command == "steer":
            instruction = str(argument or "").strip()
            if instruction:
                controller.steer(instruction)
                _control_notice(
                    session_ui,
                    f"Steering instruction recorded for {mission_id}.",
                )
            return True
    except (ValueError, OSError) as exc:
        _print(session_ui, f"[warning][Mission][/warning] {exc}")
        return True
    return False


def _handle_control_command(user_input: str, *, session_ui) -> tuple[bool, str | None]:
    """Consume /pause, /resume, /steer or /stop at the session prompt."""
    cleaned = user_input.strip()
    command, _, argument = cleaned[1:].partition(" ")
    command = command.lower()
    if command not in {"pause", "resume", "steer", "stop"}:
        return False, None
    was_paused = (
        _ACTIVE_RUNNER is not None and _ACTIVE_RUNNER.controller.status == "paused"
    )
    controlled = control_active_mission(
        command,
        argument,
        session_ui=session_ui,
    )
    if not controlled:
        if command in {"pause", "resume", "stop"}:
            _control_notice(session_ui, "No active mission.")
        else:
            _control_notice(
                session_ui,
                "/steer is available during a running turn or mission.",
            )
        return True, None
    if command == "resume" and was_paused and _ACTIVE_RUNNER is not None:
        return True, _ACTIVE_RUNNER.resume_prompt()
    return True, None


def mission_slash_command(user_input, memory, session_id, session_ui):
    """Handle /mission and live mission controls.

    Returns the next automatic mission prompt, or None when the input was
    consumed. A non-command returns unchanged.
    """
    global _ACTIVE_RUNNER
    cleaned = str(user_input or "").strip()
    lowered = cleaned.lower()
    if lowered.startswith("/") and lowered.split(maxsplit=1)[0] in {
        "/pause",
        "/resume",
        "/steer",
        "/stop",
    }:
        _consumed, next_prompt = _handle_control_command(cleaned, session_ui=session_ui)
        return next_prompt if _consumed else cleaned
    if lowered != "/mission" and not lowered.startswith("/mission "):
        return user_input
    from .mission import handle_mission_command

    result = handle_mission_command(
        memory=memory,
        session_id=session_id,
        argument=cleaned[len("/mission") :].strip(),
        session_ui=session_ui,
    )
    if result and result.get("action") == "start":
        from agent.mission_runner import MissionAutoRunner
        from agent.missions import MissionLeaseError

        if _ACTIVE_RUNNER is not None:
            _ACTIVE_RUNNER.controller.release_lease(_ACTIVE_RUNNER.owner)
            _ACTIVE_RUNNER = None
        try:
            _ACTIVE_RUNNER = MissionAutoRunner(result["controller"], session_id)
        except MissionLeaseError as exc:
            session_ui.print(f"[warning][Mission][/warning] {exc}")
            return None
        _print(
            session_ui,
            f"[info][Mission controls][/info] {mission_controls_hint()}",
        )
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
    if prompt is None and _ACTIVE_RUNNER is not None:
        mission = _ACTIVE_RUNNER.controller.mission()
        if mission["status"] not in {"paused", "waiting_user"}:
            _ACTIVE_RUNNER = None
    return prompt


def after_turn(text: str, response) -> str | None:
    """Advance the mission using final model output, not tool-call rounds."""
    return mission_after_turn(
        text,
        had_tool_calls=bool(getattr(response, "tool_calls", None)),
        interrupted=bool(getattr(response, "additional_kwargs", {}).get("interrupted")),
    )


__all__ = [
    "after_turn",
    "control_active_mission",
    "mission_after_turn",
    "mission_controls_hint",
    "mission_slash_command",
]
