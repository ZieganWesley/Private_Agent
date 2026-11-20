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
