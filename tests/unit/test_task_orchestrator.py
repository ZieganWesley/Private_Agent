from private_agent.agent import task_orchestrator


def test_build_plan_and_summary():
    plan = task_orchestrator.build_plan("goal", ["step1", "step2"])
    assert len(plan.steps) == 2
    plan.mark(0, "done")
    assert not plan.is_complete()
    plan.mark(1, "done")
    assert plan.is_complete()


def test_verify_step():
    step = task_orchestrator.TaskStep(index=0, description="run tests")
    assert task_orchestrator.verify_step(step, "all pass") is True
    step2 = task_orchestrator.TaskStep(index=0, description="research topic")
    assert task_orchestrator.verify_step(step2, "nothing here") is False
