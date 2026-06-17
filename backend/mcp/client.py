from __future__ import annotations

from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
