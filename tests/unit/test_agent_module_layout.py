"""Guards for the split of the agent runtime into focused modules."""

import inspect
import pathlib

import pytest

import private_agent.agent as agent_package
from private_agent.agent import (
    learning,
    memory_context,
    model_selection,
    runtime,
    session_state,
    streaming,
    task_commands,
    task_lease,
    tool_display,
)

AGENT_DIR = pathlib.Path(inspect.getfile(runtime)).parent


@pytest.mark.parametrize(
    ("function", "module"),
    [
        (tool_display._display_tool_result, tool_display),
        (tool_display._redact_tool_arguments, tool_display),
        (streaming._invoke_with_budget, streaming),
        (streaming._invoke_with_status, streaming),
        (model_selection.inspect_model_capabilities, model_selection),
        (model_selection.get_robust_chat_model, model_selection),
        (learning._offer_skill_capture, learning),
        (learning._run_learning_suggestion, learning),
        (memory_context._build_startup_memory_context, memory_context),
        (memory_context._run_memory_maintenance, memory_context),
        (task_commands._run_task_management, task_commands),
        (task_lease._maintain_task_lease, task_lease),
    ],
)
def test_extracted_helpers_live_in_their_own_module(function, module):
    assert inspect.getmodule(function) is module


def test_package_facade_still_resolves_every_helper():
    assert agent_package._invoke_with_budget is streaming._invoke_with_budget
    assert agent_package._run_task_management is task_commands._run_task_management
    assert agent_package._run_learning_suggestion is learning._run_learning_suggestion
    assert (
        agent_package._build_startup_memory_context
        is memory_context._build_startup_memory_context
    )
    with pytest.raises(AttributeError):
        _ = agent_package.not_a_real_helper


def test_console_and_session_state_are_shared_singletons():
    assert runtime.console is tool_display.console
    assert runtime.console is streaming.console
    assert runtime.console is task_commands.console
    assert runtime.console is memory_context.console
    assert runtime.console is learning.console
    assert runtime._SESSION_STATS is session_state.SESSION_STATS


def test_task_lease_constants_have_a_single_source():
    assert task_commands.TASK_LEASE_SECONDS == task_lease.TASK_LEASE_SECONDS
    assert task_lease.TASK_LEASE_RENEW_INTERVAL_SECONDS < task_lease.TASK_LEASE_SECONDS


def test_agent_modules_stay_below_the_previous_monolith_size():
    oversized = {
        path.name: len(path.read_text().splitlines())
        for path in AGENT_DIR.glob("*.py")
        if len(path.read_text().splitlines()) > 4000
    }
    assert oversized == {}


def test_session_startup_and_provider_selection_are_separate_functions():
    import inspect

    from private_agent.agent import session_types

    for helper in (runtime._prepare_session, runtime._choose_provider):
        assert inspect.iscoroutinefunction(helper)
        assert inspect.getmodule(helper) is runtime

    for container in (
        session_types.SessionSetup,
        session_types.ProviderSelection,
        session_types.ProviderPreference,
    ):
        assert inspect.isclass(container)


def test_session_loop_stays_small_after_the_split():
    import inspect

    session_source = inspect.getsource(runtime._run_agent_cli_session)
    # The turn loop body is still large; startup and provider selection are out.
    assert len(session_source.splitlines()) < 1900
    assert "_prepare_session(" in session_source
    assert "_choose_provider(" in session_source
