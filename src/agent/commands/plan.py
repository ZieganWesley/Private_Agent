"""Handlers for /plan and /code slash commands."""


def handle_plan_command(user_input, planning_mode_persistent, session_ui):
    """Apply a /plan command and return the updated persistent flag."""
    lowered = user_input.lower()
    if lowered in {"/plan", "/plan on"}:
        planning_mode_persistent = True
        session_ui.print(
            "[info][Planning mode][/info] Planning stays on until a "
            "plan is approved or you run /plan off."
        )
    elif lowered == "/plan off":
        planning_mode_persistent = False
        session_ui.print("[info][Planning mode off][/info]")
    return planning_mode_persistent


def handle_code_command(session_ui):
    """Arm coding mode for the next request."""
    session_ui.print(
        "[info][Coding mode][/info] Next request requires an "
        "approved coding plan before edits or commands run."
    )
    return True
