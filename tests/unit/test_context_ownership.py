"""C1/C2: the context allocation owns prompt budgets and source shares."""

from langchain_core.messages import HumanMessage

from agent import context_profiles, context_telemetry
from agent.context_engine import ContextEngine
from agent.context_manager import ContextBudget, ContextSource
from agent.context_sources import adapt_source
from agent.prompts import token_count, trim_history_messages

SOURCES = (
    "system",
    "tool_definitions",
    "skills",
    "plan",
    "files",
    "summary",
    "episodic",
    "history",
    "rag",
    "tool_results",
    "user",
)


def _engine(**kwargs) -> ContextEngine:
    return ContextEngine(enabled=True, profile="chat", **kwargs)


def test_rag_receives_its_share_instead_of_half_the_budget():
    engine = _engine()
    allocation = engine.allocate(
        1000,
        source_texts={name: "" for name in SOURCES},
        request_text="hello",
    )

    assert allocation.enabled is True
    assert sum(allocation.budgets.values()) == 1000
    # The default rag share is 0.15; it must not absorb the unused memory share.
    assert 140 <= allocation.budgets["rag"] <= 165
    assert allocation.budgets["history"] > allocation.budgets["rag"]
    assert allocation.budgets["user"] > 0


def test_budgets_sum_to_the_usable_prompt_budget():
    engine = _engine()
    allocation = engine.allocate(
        2000,
        source_texts={name: "" for name in SOURCES},
        output_reserve=500,
        safety_margin=100,
    )

    assert sum(allocation.budgets.values()) == 1400
    assert allocation.total_tokens == 2000


def test_legacy_share_names_map_onto_real_sources():
    shares = context_profiles.profile_shares(
        "chat", overrides={"recent": 0.4, "memory": 0.2}
    )

    assert "recent" not in shares
    assert "memory" not in shares
    assert shares["history"] > shares["episodic"] > 0
    assert shares["history"] > shares["summary"] > 0
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_coding_profile_gives_files_and_plan_more_than_chat():
    chat = context_profiles.profile_shares("chat")
    coding = context_profiles.profile_shares("coding")

    assert coding["files"] > chat["files"]
    assert coding["plan"] > chat["plan"]
    assert coding["history"] < chat["history"]


def test_context_shares_config_override_changes_allocation_without_code():
    engine = ContextEngine.from_config(
        {
            "agent": {
                "context_engine_enabled": True,
                "context_shares": {"rag": 0.5, "system": 0.5},
            }
        }
    )
    allocation = engine.allocate(
        1000, source_texts={name: "" for name in SOURCES}
    )

    assert allocation.budgets["rag"] >= 290
    assert allocation.budgets["rag"] > allocation.budgets["history"]
    assert allocation.budgets["rag"] > allocation.budgets["files"]
    assert allocation.budgets["rag"] > 2 * allocation.budgets["episodic"]


def test_source_constraints_are_applied():
    budget = ContextBudget(total_tokens=1000)
    sources = [
        ContextSource(name="rag", priority=10, max_tokens=50),
        ContextSource(name="history", priority=1),
    ]
    allocation = budget.allocate(sources)

    assert allocation["rag"] <= 50
    assert sum(allocation.values()) == 1000


def test_unused_budget_only_redistributes_to_compressible_sources():
    budget = ContextBudget(
        total_tokens=300,
        shares={"system": 0.5, "rag": 0.5},
        compressible={"system": False, "rag": True},
    )
    allocation = budget.allocate(["system", "rag"])
    updated = budget.redistribute_unused(
        {"system": 0, "rag": 0}, total=300
    )

    assert allocation["system"] == allocation["rag"] == 150
    assert updated["system"] == 150
    assert updated["rag"] == 150
    assert sum(updated.values()) == 300


def test_allocation_trims_each_source_to_its_budget():
    engine = _engine()
    long_text = "word " * 5000
    allocation = engine.allocate(
        1000,
        source_texts={name: (long_text if name == "rag" else "") for name in SOURCES},
    )

    rag = allocation.sources["rag"]
    assert token_count(rag.text) <= allocation.budgets["rag"]
    assert allocation.telemetry["used"]["rag"] <= allocation.budgets["rag"]
    assert allocation.telemetry["overflow"]["rag"] == 0
    assert "rag" not in allocation.telemetry["truncated"]


def test_fill_sources_keeps_budgets_and_updates_usage():
    engine = _engine()
    allocation = engine.allocate(
        1000, source_texts={name: "" for name in SOURCES}
    )
    budgets = dict(allocation.budgets)

    engine.fill_sources(
        allocation,
        {"rag": "retrieved knowledge", "user": "a question"},
    )

    assert allocation.budgets == budgets
    assert allocation.sources["rag"].text == "retrieved knowledge"
    assert allocation.telemetry["used"]["user"] == token_count("a question")
    assert sum(allocation.telemetry["used"].values()) > 0


def test_history_trimming_keeps_whole_messages_only():
    history = [
        HumanMessage(content="first " * 100),
        HumanMessage(content="second"),
        HumanMessage(content="third"),
    ]

    trimmed = trim_history_messages(history, token_count("second") + token_count("third"))

    assert [message.content for message in trimmed] == ["second", "third"]
    assert all(message in history for message in trimmed)


def test_history_trimming_drops_a_message_that_does_not_fit():
    history = [HumanMessage(content="short"), HumanMessage(content="long " * 500)]

    trimmed = trim_history_messages(history, 3)

    assert trimmed == []


def test_report_uses_the_turn_allocation_and_has_no_content():
    engine = _engine()
    allocation = engine.allocate(
        1000,
        source_texts={name: "" for name in SOURCES},
        request_text="implement a function",
    )

    report = context_telemetry.build_runtime_context_report(
        1000, [], [], "secret prompt text", "secret user text", allocation=allocation
    )

    assert report == allocation.telemetry
    assert report["sum_budgets"] == sum(allocation.budgets.values())
    assert "secret" not in repr(report)


def test_engine_disabled_keeps_the_legacy_path():
    engine = ContextEngine(enabled=False)
    allocation = engine.allocate(500, source_texts={"system": "x"})

    assert allocation.enabled is False
    assert allocation.budgets == {}
    assert allocation.sources == {}


def test_source_adapter_priorities_reach_the_budget():
    source = adapt_source("rag", text="x", budget_tokens=0)

    assert source.priority > 0
    assert source.compressible is True
