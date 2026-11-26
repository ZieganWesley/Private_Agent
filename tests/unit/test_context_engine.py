"""Unit tests for the context engine coordination layer."""

from agent import context_engine as ce


def test_from_config_defaults_disabled():
    eng = ce.ContextEngine.from_config({})
    assert eng.enabled is False


def test_from_config_enabled():
    eng = ce.ContextEngine.from_config({"agent": {"context_engine_enabled": True}})
    assert eng.enabled is True


def test_disabled_allocation_is_fallback():
    eng = ce.ContextEngine(enabled=False)
    alloc = eng.allocate(500, source_texts={"system": "x"})
    assert alloc.enabled is False
    assert alloc.budgets == {}


def test_enabled_allocation_sums_to_total():
    eng = ce.ContextEngine(enabled=True, profile="chat")
    alloc = eng.allocate(800, source_texts={"system": "s", "recent": "r" * 100})
    assert alloc.enabled is True
    assert sum(alloc.budgets.values()) == 800
    assert set(alloc.sources.keys()) == {"system", "recent"}


def test_profile_selection_in_allocation():
    eng = ce.ContextEngine(enabled=True, profile="auto")
    alloc = eng.allocate(
        400, source_texts={"files": "f"}, request_text="implement code"
    )
    assert alloc.profile == "coding"


def test_is_enabled_helper():
    assert ce.is_enabled({}) is False
    assert ce.is_enabled({"agent": {"context_engine_enabled": True}}) is True
    assert ce.is_enabled(None) is False
