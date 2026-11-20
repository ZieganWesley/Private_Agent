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


def test_classify_turn():
    packs, complexity = capabilities.classify_turn("research then implement")
    assert "Research" in packs
    assert complexity in ("simple", "multi_step", "long")


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
