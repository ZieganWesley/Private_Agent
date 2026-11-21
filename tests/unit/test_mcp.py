import pytest

from private_agent.tools.mcp import load_configured_mcp_tools


@pytest.mark.asyncio
async def test_invalid_transport_raises():
    with pytest.raises(ValueError):
        await load_configured_mcp_tools("srv", {"transport": "ftp"})


@pytest.mark.asyncio
async def test_stdio_requires_command():
    with pytest.raises(ValueError):
        await load_configured_mcp_tools("srv", {"transport": "stdio", "args": []})


@pytest.mark.asyncio
async def test_stdio_args_must_be_strings():
    with pytest.raises(ValueError):
        await load_configured_mcp_tools("srv", {"transport": "stdio", "command": "x", "args": [1]})


@pytest.mark.asyncio
async def test_stdio_env_must_be_string_map():
    with pytest.raises(ValueError):
        await load_configured_mcp_tools("srv", {"transport": "stdio", "command": "x", "env": {"A": 1}})


@pytest.mark.asyncio
async def test_timeout_must_be_in_range():
    with pytest.raises(ValueError):
        await load_configured_mcp_tools("srv", {"transport": "stdio", "command": "x", "timeout": 0})


@pytest.mark.asyncio
async def test_remote_url_must_be_valid():
    with pytest.raises(ValueError):
        await load_configured_mcp_tools("srv", {"transport": "sse", "url": "ftp://example.com"})
