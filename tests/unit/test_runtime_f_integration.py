from private_agent import runtime_f_integration


def test_is_plan_complete_steers():
    state = runtime_f_integration.begin_turn("t1", "do thing", 100)
    assert not runtime_f_integration.is_plan_complete(state)
    runtime_f_integration.finish_iteration(state, 0, 0.0, "ok", True)
    assert runtime_f_integration.is_plan_complete(state)
