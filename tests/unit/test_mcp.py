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
