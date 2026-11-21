import pytest

from private_agent import capabilities


@pytest.fixture(autouse=True)
def _clear_enabled_packs():
    from private_agent.tools import get_tool_context

    get_tool_context().enabled_packs.clear()
    yield
    get_tool_context().enabled_packs.clear()


def test_route_packs_explicit_code():
    packs = capabilities.route_packs("/code fix bug")
    assert "Coding" in packs


def test_route_packs_url_adds_research():
    packs = capabilities.route_packs("check https://example.com")
    assert "Research" in packs


def test_route_packs_attachment_adds_coding():
    packs = capabilities.route_packs("explain", has_attachment=True)
    assert "Coding" in packs


def test_tools_for_packs():
    tools = capabilities.tools_for_packs({"Core"})
    assert "read_local_file" in tools
    assert "get_local_datetime" in tools
    assert "web_search" not in tools


def test_tools_for_packs_research():
    tools = capabilities.tools_for_packs({"Core", "Research"})
    assert "web_search" in tools


def test_enable_capability():
    assert "Enabled" in capabilities.enable_capability("Research")
    assert "Unknown" in capabilities.enable_capability("Nope")


def test_filter_preserves_unlisted_tools():
    from types import SimpleNamespace

    tools = [
        SimpleNamespace(name="web_search"),
        SimpleNamespace(name="totally_unknown_tool"),
    ]
    out = capabilities.filter_tools_list(tools, "hello")
    names = {t.name for t in out}
    assert "totally_unknown_tool" in names


def test_all_registered_tools_preserved():
    from types import SimpleNamespace

    import private_agent.tools as tools_mod

    names = list(tools_mod.AVAILABLE_TOOLS.keys())
    objs = [SimpleNamespace(name=n) for n in names]
    out = capabilities.filter_tools_list(
        objs,
        "/code /research /plan fix and search https://example.com weather image "
        "chat history record audio",
    )
    out_names = {t.name for t in out}
    assert set(names).issubset(out_names)


def test_every_registered_tool_belongs_to_a_pack():
    """Tools outside every pack are always sent, defeating J4 filtering."""
    import private_agent.tools as tools_mod

    packed = capabilities._all_pack_tools()
    assert set(tools_mod.AVAILABLE_TOOLS) - packed == set()


def test_plain_chat_binds_only_core_tools():
    from types import SimpleNamespace

    import private_agent.tools as tools_mod

    objs = [SimpleNamespace(name=n) for n in tools_mod.AVAILABLE_TOOLS]
    out = {tool.name for tool in capabilities.filter_tools_list(objs, "hello there")}
    assert out == set(capabilities.CORE_TOOLS)
    assert "web_search" not in out
    assert "edit_local_file" not in out


def test_route_packs_media_memory_and_helper_keywords():
    assert "Media" in capabilities.route_packs("describe this image")
    assert "Memory" in capabilities.route_packs("what did we discuss earlier?")
    assert "Memory" in capabilities.route_packs("remember this for later")
    assert "Helper" in capabilities.route_packs("what is the weather")
    assert "Helper" in capabilities.route_packs("make me a pdf")


def test_filter_tools_list_is_sorted_for_a_stable_prefix():
    from types import SimpleNamespace

    tools = [
        SimpleNamespace(name="web_search"),
        SimpleNamespace(name="ask_user"),
        SimpleNamespace(name="get_local_datetime"),
    ]
    out = capabilities.filter_tools_list(tools, "search the web")
    assert [tool.name for tool in out] == [
        "ask_user",
        "get_local_datetime",
        "web_search",
    ]


def test_gate_capability_non_gated_pack():
    from types import SimpleNamespace

    import private_agent.runtime_f_integration as f

    state = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console = SimpleNamespace(input=lambda *a, **k: "n")
    assert (
        f.gate_capability({"args": {"name": "Planning"}}, "auto", console, state)
        is True
    )
    assert state.capability_approved is True
    assert state.blocked_tool_messages == []


def test_gate_capability_coding_denied_noninteractive():
    from types import SimpleNamespace

    import private_agent.runtime_f_integration as f

    state = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console = SimpleNamespace(input=lambda *a, **k: "y")
    assert (
        f.gate_capability({"args": {"name": "Coding"}}, "auto", console, state) is False
    )
    assert state.capability_approved is False
    assert state.blocked_tool_messages


def test_gate_capability_coding_approved_interactive(monkeypatch):
    from types import SimpleNamespace

    import private_agent.runtime_f_integration as f

    # Force interactive tty
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    import private_agent.tools as tools_mod

    monkeypatch.setattr(tools_mod.get_tool_context(), "interactive", True)

    state = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console = SimpleNamespace(input=lambda *a, **k: "y")
    assert (
        f.gate_capability({"args": {"name": "Coding"}}, "auto", console, state) is True
    )
    assert state.capability_approved is True

    state2 = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console2 = SimpleNamespace(input=lambda *a, **k: "n")
    assert (
        f.gate_capability({"args": {"name": "Research"}}, "manual", console2, state2)
        is False
    )
    assert state2.blocked_tool_messages


def test_every_pack_tool_name_is_registered():
    """A pack that names an unregistered tool can never gate the real one.

    ``filter_tools_list`` preserves tools no pack mentions, so a stale pack name
    silently leaves the actual tool always available. Derive the registry from
    the real tool objects rather than a hand-maintained list so a rename on
    either side fails here.
    """
    import private_agent.tools as tools_mod
    from private_agent import code_tasks

    # Code-task tools are injected into AVAILABLE_TOOLS only for code-task turns.
    registered = set(tools_mod.AVAILABLE_TOOLS)
    registered.add(code_tasks.run_project_unit_tests.name)

    assert capabilities.unknown_pack_tools(registered) == {}
