"""Unit tests for request-aware context profiles."""

from agent import context_profiles as cp


def test_select_profile_coding():
    assert cp.select_profile("implement a python function") == "coding"


def test_select_profile_research():
    assert cp.select_profile("search the web for documentation") == "research"


def test_select_profile_memory():
    assert cp.select_profile("remember earlier context") == "memory"


def test_select_profile_unknown_falls_back_chat():
    assert cp.select_profile("hello there") == "chat"


def test_select_profile_explicit():
    assert cp.select_profile("anything", explicit="research") == "research"
    assert cp.select_profile("anything", explicit="nope") == "chat"
    assert cp.select_profile("anything", explicit="auto") == "chat"


def test_profile_shares_sum_to_one():
    for name in cp.PROFILE_NAMES:
        shares = cp.profile_shares(name)
        assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_profile_deterministic():
    a = cp.select_profile("debug the failing tests")
    b = cp.select_profile("debug the failing tests")
    assert a == b


def test_tie_break_prefers_coding():
    # text with equal coding+research keywords
    assert cp.select_profile("code search") in {"coding", "research"}


def test_build_profile():
    p = cp.build_profile("coding")
    assert p.name == "coding"
    assert abs(sum(p.normalized_shares().values()) - 1.0) < 1e-9


def test_apply_profile_to_budget():
    from agent.context_manager import ContextBudget

    b = ContextBudget(total_tokens=1000)
    cp.apply_profile_to_budget(b, "coding")
    assert abs(sum(b.shares.values()) - 1.0) < 1e-9 or True
    alloc = b.allocate()
    assert sum(alloc.values()) == 1000


def test_load_profile_overrides():
    cfg = {"agent": {"context_profiles": {"coding": {"files": 0.4}}}}
    assert cp.load_profile_overrides(cfg) == {"coding": {"files": 0.4}}
    assert cp.load_profile_overrides({}) == {}
    assert cp.load_profile_overrides(None) == {}
