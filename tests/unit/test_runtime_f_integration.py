from private_agent.agent import runtime_f_integration


def test_is_plan_complete_steers():
    state = runtime_f_integration.begin_turn("t1", "do thing", 100)
    assert not runtime_f_integration.is_plan_complete(state)
    assert runtime_f_integration.get_plan_summary(state)
    runtime_f_integration.finish_iteration(state, 0, 0.0, "ok", True)
    assert runtime_f_integration.is_plan_complete(state)
    runtime_f_integration.finish_turn(state)
    assert runtime_f_integration.is_plan_complete(state)


def test_begin_turn_isolates_and_restores_enabled_packs():
    from private_agent.tools import get_tool_context

    outer = get_tool_context()
    outer.enabled_packs.add("Research")
    try:
        state = runtime_f_integration.begin_turn("t1", "do thing", 100)
        assert get_tool_context().enabled_packs == set()
        get_tool_context().enabled_packs.add("Coding")
    finally:
        runtime_f_integration.finish_turn(state)
    assert get_tool_context().enabled_packs == {"Research"}
    outer.enabled_packs.clear()
