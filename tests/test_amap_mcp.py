from __future__ import annotations

import asyncio

from backend.mcp import amap
from backend.mcp.amap import should_use_amap_mcp
from backend.mcp.client import AMAP_MCP_URL, build_amap_mcp_client
from backend.schemas.trip import McpQuery, McpQueryPlan, McpQueryStage, McpResults, McpToolName, RouteResult, TransportMode


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


def test_amap_query_plan_keeps_successful_results_when_one_query_fails(monkeypatch) -> None:
    async def load_tools(provider=None):
        return []

    async def execute_query(query, default_city, tools):
        if query.args["origin"] == "bad":
            raise RuntimeError("OVER_DIRECTION_RANGE")
        return McpResults(
            routes=[
                RouteResult(
                    origin=query.args["origin"],
                    destination=query.args["destination"],
                    origin_city=query.args["origin_city"],
                    destination_city=query.args["destination_city"],
                    mode=TransportMode(query.args["mode"]),
                    duration_minutes=30,
                    distance_km=10,
                )
            ]
        )

    monkeypatch.setattr(amap, "load_amap_tools", load_tools)
    monkeypatch.setattr(amap, "_execute_amap_mcp_query", execute_query)

    query_plan = McpQueryPlan(
        queries=[
            McpQuery(
                tool_name=McpToolName.GET_ROUTE_TIME,
                args={
                    "origin": "good",
                    "destination": "target",
                    "origin_city": "Good City",
                    "destination_city": "Target City",
                    "mode": "taxi",
                },
                purpose="successful query",
                stage=McpQueryStage.PLAN_CHECK,
            ),
            McpQuery(
                tool_name=McpToolName.GET_ROUTE_TIME,
                args={
                    "origin": "bad",
                    "destination": "target",
                    "origin_city": "Bad City",
                    "destination_city": "Target City",
                    "mode": "taxi",
                },
                purpose="failing query",
                stage=McpQueryStage.PLAN_CHECK,
            ),
        ]
    )

    result = asyncio.run(amap._execute_amap_mcp_query_plan(query_plan, default_city="Target City"))

    assert len(result.routes) == 1
    assert result.routes[0].origin == "good"
    assert len(result.errors) == 1
    assert "OVER_DIRECTION_RANGE" in result.errors[0]
