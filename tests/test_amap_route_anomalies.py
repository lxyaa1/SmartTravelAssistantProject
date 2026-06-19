from __future__ import annotations

import os

import pytest

from backend.mcp.amap import execute_amap_mcp_query_plan
from backend.mcp.client import get_amap_api_key, get_bailian_api_key
from backend.schemas.trip import McpQuery, McpQueryPlan, McpQueryStage, McpToolName


ROUTE_CASES = [
    ("太原", "五台山", "transit"),
    ("五台山", "太原", "transit"),
    ("太原", "北京", "train"),
    ("北京", "太原", "train"),
]


def test_live_amap_route_anomaly_cases() -> None:
    """Diagnostic live test for route values that produced bad plans.

    Run with:
      set RUN_AMAP_MCP_LIVE_TESTS=1
      set TRAVEL_AGENT_AMAP_PROVIDER=official
      python -m pytest tests/test_amap_route_anomalies.py -q -s

    The assertions intentionally encode broad sanity checks, not exact route
    truth. If this fails, the workflow should not trust that MCP route result.
    """
    _require_live_amap_tests()

    query_plan = McpQueryPlan(
        queries=[
            McpQuery(
                tool_name=McpToolName.GET_ROUTE_TIME,
                args={"origin": origin, "destination": destination, "mode": mode},
                purpose="diagnose abnormal Amap route result",
                stage=McpQueryStage.PLAN_CHECK,
            )
            for origin, destination, mode in ROUTE_CASES
        ]
    )

    results = execute_amap_mcp_query_plan(
        query_plan=query_plan,
        default_city="太原",
        provider=os.getenv("TRAVEL_AGENT_AMAP_PROVIDER", "official"),
    )
    routes = {(route.origin, route.destination, route.mode.value): route for route in results.routes}

    print("\nAmap normalized route results:")
    for origin, destination, mode in ROUTE_CASES:
        route = routes[(origin, destination, mode)]
        print(
            f"- {origin} -> {destination} ({mode}): "
            f"{route.duration_minutes} min, {route.distance_km} km"
        )

    taiyuan_to_wutai = routes[("太原", "五台山", "transit")]
    wutai_to_taiyuan = routes[("五台山", "太原", "transit")]
    taiyuan_to_beijing = routes[("太原", "北京", "train")]
    beijing_to_taiyuan = routes[("北京", "太原", "train")]

    assert 100 <= taiyuan_to_wutai.distance_km <= 300
    assert 100 <= wutai_to_taiyuan.distance_km <= 300
    assert 300 <= taiyuan_to_beijing.distance_km <= 700
    assert 300 <= beijing_to_taiyuan.distance_km <= 700

    assert _distance_ratio(taiyuan_to_wutai.distance_km, wutai_to_taiyuan.distance_km) <= 1.5
    assert _distance_ratio(taiyuan_to_beijing.distance_km, beijing_to_taiyuan.distance_km) <= 1.5


def _require_live_amap_tests() -> None:
    if os.getenv("RUN_AMAP_MCP_LIVE_TESTS") != "1":
        pytest.skip("set RUN_AMAP_MCP_LIVE_TESTS=1 to call live Amap MCP tools")

    provider = os.getenv("TRAVEL_AGENT_AMAP_PROVIDER", "official").strip().lower()
    if provider in {"official", "amap_official", "auto"} and get_amap_api_key():
        return
    if provider in {"bailian", "dashscope", "amap_bailian", "auto"} and get_bailian_api_key():
        return
    pytest.skip("set Amap_Key for official Amap MCP or DASHSCOPE_API_KEY for Bailian Amap MCP")


def _distance_ratio(first: float, second: float) -> float:
    smaller = max(0.001, min(first, second))
    return max(first, second) / smaller
