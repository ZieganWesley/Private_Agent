from private_agent import capabilities


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
    assert "web_search" not in tools


def test_tools_for_packs_research():
    tools = capabilities.tools_for_packs({"Core", "Research"})
    assert "web_search" in tools


def test_enable_capability():
    assert "Enabled" in capabilities.enable_capability("Research")
    assert "Unknown" in capabilities.enable_capability("Nope")


def test_filter_preserves_unlisted_tools():
    from types import SimpleNamespace
    tools = [SimpleNamespace(name="web_search"), SimpleNamespace(name="totally_unknown_tool")]
    out = capabilities.filter_tools_list(tools, "hello")
    names = {t.name for t in out}
    assert "totally_unknown_tool" in names


def test_all_registered_tools_preserved():
    from types import SimpleNamespace

    import private_agent.tools as tools_mod
    names = list(tools_mod.AVAILABLE_TOOLS.keys())
    objs = [SimpleNamespace(name=n) for n in names]
    out = capabilities.filter_tools_list(objs, "/code /research /plan fix and search https://example.com")
    out_names = {t.name for t in out}
    assert set(names).issubset(out_names)


def test_gate_capability_non_gated_pack():
    from types import SimpleNamespace

    import private_agent.runtime_f_integration as f

    state = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console = SimpleNamespace(input=lambda *a, **k: "n")
    assert f.gate_capability({"args": {"name": "Planning"}}, "auto", console, state) is True
    assert state.capability_approved is True
    assert state.blocked_tool_messages == []


def test_gate_capability_coding_denied_noninteractive():
    from types import SimpleNamespace

    import private_agent.runtime_f_integration as f

    state = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console = SimpleNamespace(input=lambda *a, **k: "y")
    assert f.gate_capability({"args": {"name": "Coding"}}, "auto", console, state) is False
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
    assert f.gate_capability({"args": {"name": "Coding"}}, "auto", console, state) is True
    assert state.capability_approved is True

    state2 = SimpleNamespace(blocked_tool_messages=[], capability_approved=False)
    console2 = SimpleNamespace(input=lambda *a, **k: "n")
    assert f.gate_capability({"args": {"name": "Research"}}, "manual", console2, state2) is False
    assert state2.blocked_tool_messages
