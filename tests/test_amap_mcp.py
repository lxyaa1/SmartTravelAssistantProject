from __future__ import annotations

from backend.mcp.amap import should_use_amap_mcp
from backend.mcp.client import AMAP_MCP_URL, build_amap_mcp_client


def test_amap_mcp_url_points_to_dashscope_bailian_server() -> None:
    assert AMAP_MCP_URL == "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"


def test_should_use_amap_mcp_can_be_forced_by_state() -> None:
    assert should_use_amap_mcp({"mcp_backend": "amap"}) is True
    assert should_use_amap_mcp({"mcp_backend": "mock"}) is False


def test_should_use_amap_mcp_uses_env_backend(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_MCP_BACKEND", "amap")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    assert should_use_amap_mcp({}) is True


def test_build_amap_mcp_client_accepts_dashscope_api_key() -> None:
    client = build_amap_mcp_client("test-key")

    assert client is not None
