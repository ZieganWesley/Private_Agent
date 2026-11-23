"""Handlers for thinking-related slash commands."""


def handle_think_toggle(
    *,
    command,
    selected_model,
    selected_local_runtime,
    thinking_effort,
    capabilities,
    tools_list,
    session_ui,
    make_model,
    bind_tools,
):
    """Toggle thinking and return (thinking_enabled, llm, llm_with_tools)."""
    requested = command.split()
    thinking_enabled = requested[-1] == "on"
    if capabilities.get("thinking") is not True:
        session_ui.print(
            f"[warning]Model '{selected_model}' does not declare thinking support; "
            "the requested setting is not active.[/warning]"
        )
        return thinking_enabled, None, None
    llm = make_model(
        selected_local_runtime,
        selected_model,
        thinking_enabled=thinking_enabled,
        thinking_effort=thinking_effort,
        capabilities=capabilities,
    )
    llm_with_tools = (
        bind_tools(llm, tools_list)
        if capabilities.get("tools") is not False
        else llm
    )
    session_ui.print(
        f"[info]Thinking: {'on' if thinking_enabled else 'off'}[/info]"
    )
    return thinking_enabled, llm, llm_with_tools


def handle_think_status(
    *,
    selected_model,
    thinking_enabled,
    thinking_effort,
    capabilities,
    session_ui,
):
    effective_thinking = thinking_enabled and capabilities.get("thinking") is True
    session_ui.print(
        f"[info]Model: {selected_model}; thinking: "
        f"{'on' if effective_thinking else 'off'}; requested: "
        f"{'on' if thinking_enabled else 'off'}; effort: {thinking_effort}; "
        f"supported: {capabilities.get('thinking')}[/info]"
    )


def handle_hardware_status(
    *,
    provider_type,
    selected_local_runtime,
    session_ui,
):
    if provider_type == "local" and selected_local_runtime is not None:
        session_ui.print(
            f"[info][Hardware Status][/info] "
            f"{selected_local_runtime.name} does not expose a "
            "standardized hardware-placement API. Check that "
            "runtime's own diagnostics for backend and device use."
        )
    else:
        session_ui.print(
            "[info][Hardware Status][/info] Select a local model "
            "to view runner-specific hardware diagnostics."
        )
