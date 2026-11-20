from private_agent import runtime_f_integration


def test_is_plan_complete_steers():
    state = runtime_f_integration.begin_turn("t1", "do thing", 100)
    assert not runtime_f_integration.is_plan_complete(state)
    assert runtime_f_integration.get_plan_summary(state)
    runtime_f_integration.finish_iteration(state, 0, 0.0, "ok", True)
    assert runtime_f_integration.is_plan_complete(state)
    runtime_f_integration.finish_turn(state)
    assert runtime_f_integration.is_plan_complete(state)


def test_begin_turn_resets_enabled_packs():
    from private_agent.tools import get_tool_context

    get_tool_context().enabled_packs.add("Coding")
    state = runtime_f_integration.begin_turn("t1", "do thing", 100)
    assert get_tool_context().enabled_packs == set()
    runtime_f_integration.finish_turn(state)
