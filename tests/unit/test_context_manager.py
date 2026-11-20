from private_agent import context_manager


def test_allocate_sums_to_total():
    b = context_manager.ContextBudget(total_tokens=1000)
    alloc = b.allocate()
    assert sum(alloc.values()) == 1000


def test_build_context_trims():
    out = context_manager.build_context({"recent": "a" * 10000}, 100)
    assert len(out["recent"]) <= 100 * 4 + 5
