from __future__ import annotations

import os

import pytest

from backend.mcp.amap import execute_amap_mcp_query_plan
from backend.mcp.client import get_amap_api_key, get_bailian_api_key
from backend.schemas.trip import McpQuery, McpQueryPlan, McpQueryStage, McpToolName


ROUTE_CASES = [
    {
        "origin": "Taiyuan",
        "destination": "Wutai Mountain",
        "origin_city": "Taiyuan",
        "destination_city": "Xinzhou",
        "mode": "transit",
        "distance_range": (100, 300),
        "duration_range": (120, 500),
    },
    {
        "origin": "Wutai Mountain",
        "destination": "Taiyuan",
        "origin_city": "Xinzhou",
        "destination_city": "Taiyuan",
        "mode": "transit",
        "distance_range": (100, 300),
        "duration_range": (120, 500),
    },
    {
        "origin": "Taiyuan",
        "destination": "Beijing",
        "origin_city": "Taiyuan",
        "destination_city": "Beijing",
        "mode": "train",
        "distance_range": (300, 700),
        "duration_range": (180, 600),
    },
    {
        "origin": "Beijing",
        "destination": "Taiyuan",
        "origin_city": "Beijing",
        "destination_city": "Taiyuan",
        "mode": "train",
        "distance_range": (300, 700),
        "duration_range": (180, 600),
    },
]


def test_live_amap_route_anomaly_cases_with_city_context() -> None:
    """Run manually to verify Amap route values for previously bad cases.

    PowerShell:
      $env:RUN_AMAP_MCP_LIVE_TESTS="1"
      $env:TRAVEL_AGENT_AMAP_PROVIDER="official"
      python -m pytest tests/test_amap_route_anomalies.py -q -s
    """
    _require_live_amap_tests()

    query_plan = McpQueryPlan(
        queries=[
            McpQuery(
                tool_name=McpToolName.GET_ROUTE_TIME,
                args={key: value for key, value in case.items() if key not in {"distance_range", "duration_range"}},
                purpose="diagnose Amap route result with explicit endpoint cities",
                stage=McpQueryStage.PLAN_CHECK,
            )
            for case in ROUTE_CASES
        ]
    )

    results = execute_amap_mcp_query_plan(
        query_plan=query_plan,
        default_city="Taiyuan",
        provider=os.getenv("TRAVEL_AGENT_AMAP_PROVIDER", "official"),
    )
    routes = {
        (route.origin, route.destination, route.origin_city, route.destination_city, route.mode.value): route
        for route in results.routes
    }

    print("\nAmap normalized route results:")
    for case in ROUTE_CASES:
        route = routes[
            (
                case["origin"],
                case["destination"],
                case["origin_city"],
                case["destination_city"],
                case["mode"],
            )
        ]
        print(
            f"- {route.origin}({route.origin_city}) -> "
            f"{route.destination}({route.destination_city}) "
            f"({route.mode.value}): {route.duration_minutes} min, {route.distance_km} km"
        )
        low, high = case["distance_range"]
        assert low <= route.distance_km <= high
        duration_low, duration_high = case["duration_range"]
        assert duration_low <= route.duration_minutes <= duration_high

    taiyuan_to_wutai = routes[("Taiyuan", "Wutai Mountain", "Taiyuan", "Xinzhou", "transit")]
    wutai_to_taiyuan = routes[("Wutai Mountain", "Taiyuan", "Xinzhou", "Taiyuan", "transit")]
    taiyuan_to_beijing = routes[("Taiyuan", "Beijing", "Taiyuan", "Beijing", "train")]
    beijing_to_taiyuan = routes[("Beijing", "Taiyuan", "Beijing", "Taiyuan", "train")]

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
