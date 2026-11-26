"""Provider loss handling and mid-turn fallback (Part I, I1o)."""

from __future__ import annotations

from utils.run_logging import RUN_LOGGER, log_event

from .model_selection import (
    _discover_available_local_runtimes,
    _make_local_openai_compatible_chat_model,
    _make_selected_local_chat_model,
    inspect_openai_compatible_model_capabilities,
)
from .provider_resume import persist_provider_choice, provider_loss_guidance
from .providers import online_request_failure as _online_request_failure
from .streaming import _invoke_with_status

_TOOL_UNSUPPORTED_MARKERS = (
    "does not support tools",
    "tool calling is not supported",
    "tools are not supported",
)


def _bind_tools(llm, tools_list):
    """Bind the active tool set to an LLM (B1 step 1 helper)."""
    return llm.bind_tools(tools_list)


def next_local_provider(*, exclude_name, discover=None):
    """Return ``(runtime, model, capabilities)`` for another reachable provider."""
    try:
        runtimes = (discover or _discover_available_local_runtimes)()
    except OSError, RuntimeError:
        return None
    excluded = str(exclude_name or "").strip().lower()
    for runtime in runtimes:
        name = str(getattr(runtime, "name", "")).strip().lower()
        if not name or name == excluded:
            continue
        models = tuple(getattr(runtime, "models", ()) or ())
        if not models:
            continue
        model = str(models[0])
        try:
            capabilities = inspect_openai_compatible_model_capabilities(runtime, model)
        except (OSError, RuntimeError, ValueError) as exc:
            log_event(
                RUN_LOGGER,
                "provider.fallback_capabilities_failed",
                provider=name,
                error_type=type(exc).__name__,
            )
            capabilities = {}
        return runtime, model, capabilities
    return None


async def _recover_on_another_provider(
    *,
    selected_local_runtime,
    tools_list,
    messages,
    deadline,
    thinking_enabled,
    thinking_effort,
    usage,
    console,
    memory,
    session_id,
    interactive,
    input_fn,
    discover,
    first_local_turn: bool = False,
    cancel_event=None,
):
    """Explain the loss, offer another provider and retry the turn on it."""
    console.print(
        provider_loss_guidance(
            runtime_name=selected_local_runtime.name,
            session_id=session_id or "<session id>",
            discover=discover or _discover_available_local_runtimes,
        )
    )
    candidate = next_local_provider(
        exclude_name=selected_local_runtime.name, discover=discover
    )
    if candidate is None or not interactive:
        return None
    runtime, model, capabilities = candidate
    ask = input_fn or console.input
    try:
        answer = ask(
            f"[info]Retry this turn on {runtime.name} · {model}? [y/N]: [/info]"
        )
    except EOFError, KeyboardInterrupt:
        return None
    if str(answer).strip().lower() not in {"y", "yes"}:
        return None
    try:
        llm = _make_selected_local_chat_model(
            runtime,
            model,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
            capabilities=capabilities,
        )
        llm_with_tools = _bind_tools(llm, tools_list)
        response = await _invoke_with_status(
            llm_with_tools,
            messages,
            deadline,
            model_name=model,
            status_message="Generating response...",
            local_model=True,
            first_local_turn=first_local_turn,
            thinking_enabled=thinking_enabled,
            context_usage=usage,
            cancel_event=cancel_event,
        )
    except Exception as exc:  # noqa: BLE001 - a second provider may fail too; report it as a failed turn
        console.print(
            f"[error][Provider fallback] {runtime.name} also failed "
            f"({type(exc).__name__}); the turn was not completed.[/error]"
        )
        return None
    persist_provider_choice(
        console=console,
        memory=memory,
        session_id=session_id,
        runtime=runtime,
        model=model,
        capabilities=capabilities,
        previous=None,
    )
    console.print(
        f"[success][Provider switched][/success] Continuing this session on "
        f"{runtime.name} · {model}."
    )
    return llm, llm_with_tools, model, capabilities, response, runtime


async def _invoke_initial_model_with_fallback(
    *,
    llm,
    llm_with_tools,
    tools_list,
    messages,
    deadline,
    selected_model,
    provider_type,
    selected_local_runtime,
    available_models,
    capabilities,
    thinking_enabled,
    thinking_effort,
    status_message,
    usage,
    console,
    memory=None,
    session_id="",
    interactive=False,
    input_fn=None,
    discover=None,
    first_local_turn: bool = True,
    cancel_event=None,
):
    """Invoke the model, falling back to a tool-capable or other local provider."""
    try:
        response = await _invoke_with_status(
            llm_with_tools,
            messages,
            deadline,
            model_name=selected_model,
            status_message=status_message,
            local_model=provider_type == "local",
            first_local_turn=first_local_turn,
            thinking_enabled=thinking_enabled,
            context_usage=usage,
            cancel_event=cancel_event,
        )
        return (
            llm,
            llm_with_tools,
            selected_model,
            capabilities,
            response,
            selected_local_runtime,
        )
    except Exception as e:  # noqa: BLE001 - inspects any model error to detect missing tool support, then re-raises
        error_text = str(e).lower()
        tool_unsupported = any(
            marker in error_text for marker in _TOOL_UNSUPPORTED_MARKERS
        )
        fallback_model = (
            next(
                (
                    candidate
                    for candidate in available_models
                    if candidate != selected_model
                    and inspect_openai_compatible_model_capabilities(
                        selected_local_runtime, candidate
                    )["tools"]
                    is True
                ),
                None,
            )
            if (
                tool_unsupported
                and provider_type == "local"
                and selected_local_runtime is not None
            )
            else None
        )
        if fallback_model is None:
            if provider_type == "online":
                raise RuntimeError(_online_request_failure(e)) from None
            recovered = None
            if selected_local_runtime is not None:
                recovered = await _recover_on_another_provider(
                    selected_local_runtime=selected_local_runtime,
                    tools_list=tools_list,
                    messages=messages,
                    deadline=deadline,
                    thinking_enabled=thinking_enabled,
                    thinking_effort=thinking_effort,
                    usage=usage,
                    console=console,
                    memory=memory,
                    session_id=session_id,
                    interactive=interactive,
                    input_fn=input_fn,
                    discover=discover,
                    first_local_turn=first_local_turn,
                    cancel_event=cancel_event,
                )
            if recovered is not None:
                return recovered
            raise RuntimeError(f"Local model request failed: {e}") from e
        console.print(
            f"[warning][Model fallback] {selected_model} rejected tool use; "
            f"retrying with {fallback_model}.[/warning]"
        )
        selected_model = fallback_model
        capabilities = inspect_openai_compatible_model_capabilities(
            selected_local_runtime, selected_model
        )
        llm = _make_local_openai_compatible_chat_model(
            selected_local_runtime.base_url,
            selected_model,
        )
        llm_with_tools = _bind_tools(llm, tools_list)
        response = await _invoke_with_status(
            llm_with_tools,
            messages,
            deadline,
            model_name=selected_model,
            status_message="Generating response...",
            local_model=provider_type == "local",
            first_local_turn=first_local_turn,
            thinking_enabled=thinking_enabled,
            cancel_event=cancel_event,
        )
        return (
            llm,
            llm_with_tools,
            selected_model,
            capabilities,
            response,
            selected_local_runtime,
        )
