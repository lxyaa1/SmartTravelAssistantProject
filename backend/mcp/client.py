from __future__ import annotations

from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMAP_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"


def build_mock_mcp_client() -> MultiServerMCPClient:
    """Create a local stdio MCP client config for future tool-backed nodes."""
    server_path = PROJECT_ROOT / "mcp_servers" / "mock_travel_server" / "server.py"
    return MultiServerMCPClient(
        {
            "mock_travel": {
                "command": "python",
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )


async def load_mock_tools():
    client = build_mock_mcp_client()
    return await client.get_tools()


def build_amap_mcp_client(api_key: str) -> MultiServerMCPClient:
    """Create a Streamable HTTP MCP client for Bailian's official Amap Maps server."""
    return MultiServerMCPClient(
        {
            "amap_maps": {
                "url": AMAP_MCP_URL,
                "headers": {"Authorization": f"Bearer {api_key}"},
                "transport": "streamable_http",
            }
        }
    )


async def load_amap_tools(api_key: str):
    client = build_amap_mcp_client(api_key)
    return await client.get_tools()
