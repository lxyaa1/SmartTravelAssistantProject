from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from langchain_core.tools import ToolException

from backend.mcp.amap import _extract_pois
from backend.mcp.client import get_amap_api_key, get_bailian_api_key, load_amap_tools, load_mock_tools
from backend.mcp.tool_registry import AMAP_MAPS_TOOLS, MOCK_TRAVEL_TOOLS
from mcp_servers.mock_travel_server.server import (
    get_attraction_detail,
    get_route_time,
    get_weather,
    search_accommodation_areas,
    search_attractions,
    search_lodging_near_place,
)


AMAP_TEST_ORIGIN = "120.155070,30.274085"
AMAP_TEST_DESTINATION = "120.168057,30.263363"
AMAP_TEST_CITY = "\u676d\u5dde"
AMAP_TEST_POI = "\u897f\u6e56"
AMAP_TEST_STATION = "\u676d\u5dde\u7ad9"
AMAP_TEST_AROUND_KEYWORD = "\u5496\u5561"


def _run(coro):
    return asyncio.run(coro)


def _require_live_amap_tests() -> tuple[str, str]:
    if os.getenv("RUN_AMAP_MCP_LIVE_TESTS") != "1":
        pytest.skip("set RUN_AMAP_MCP_LIVE_TESTS=1 to call live Amap MCP tools")

    provider = os.getenv("TRAVEL_AGENT_AMAP_PROVIDER", "auto").strip().lower()
    amap_key = get_amap_api_key()
    dashscope_key = get_bailian_api_key()
    if provider in {"official", "amap_official"}:
        if not amap_key:
            pytest.skip("Amap_Key, AMAP_KEY, or AMAP_MAPS_API_KEY is not set")
        return "official", amap_key
    if provider in {"bailian", "dashscope", "amap_bailian"}:
        if not dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY is not set")
        return "bailian", dashscope_key
    if amap_key:
        return "official", amap_key
    if dashscope_key:
        return "bailian", dashscope_key
    pytest.skip("no Amap MCP key is set")


def _assert_tool_output(tool_name: str, result: Any) -> None:
    assert result is not None, f"{tool_name} returned None"
    text = str(result).strip()
    assert text, f"{tool_name} returned an empty response"
    assert "INVALID_USER_KEY" not in text, f"{tool_name} returned INVALID_USER_KEY"
    assert "API" not in text.upper() or "FAILED" not in text.upper(), text[:500]


async def _invoke_amap_tool(tool, payload: dict[str, Any], tool_name: str) -> Any:
    try:
        return await tool.ainvoke(payload)
    except ToolException as exc:
        message = str(exc)
        if "INVALID_USER_KEY" in message:
            pytest.skip(f"Amap MCP tool calls are not authorized for this key yet: {message}")
        raise AssertionError(f"{tool_name} failed: {message}") from exc


def test_mock_mcp_tools_are_discoverable_over_stdio() -> None:
    tools = _run(load_mock_tools())
    tool_names = {tool.name for tool in tools}

    assert set(MOCK_TRAVEL_TOOLS.tool_names).issubset(tool_names)


def test_mock_weather_contains_outdoor_conflict() -> None:
    result = get_weather("Hangzhou", "2026-07-01")

    assert result["condition"] == "heavy rain"
    assert result["warning"] == "Outdoor plans may be affected."


def test_mock_attraction_detail_contains_closure_conflict() -> None:
    result = get_attraction_detail("Hangzhou Museum", "2026-07-02")

    assert result["is_open"] is False
    assert result["opening_hours"] == "Closed"


def test_mock_route_time_contains_long_route_conflict() -> None:
    result = get_route_time("West Lake", "Hangzhou Museum", "taxi")

    assert result["duration_minutes"] == 150
    assert result["distance_km"] == 40


def test_mock_search_attractions_returns_outdoor_and_indoor_options() -> None:
    results = search_attractions("Hangzhou", ["culture", "family friendly"])
    categories = {item["category"] for item in results}

    assert "outdoor" in categories
    assert "indoor" in categories


def test_mock_search_accommodation_areas_returns_area_options() -> None:
    results = search_accommodation_areas("Hangzhou", "medium", True)

    assert results
    assert results[0]["area_name"] == "Hangzhou Lakeside Area"
    assert "families" in results[0]["suitable_for"]


def test_mock_search_lodging_near_place_returns_anchor_lodging() -> None:
    results = search_lodging_near_place("Hangzhou", "West Lake", "medium", True, 5)

    assert results
    assert results[0]["name"] == "West Lake Nearby Hotel"
    assert results[0]["anchor_place"] == "West Lake"
    assert results[0]["duration_to_anchor_minutes"] == 12


def test_amap_mcp_discovers_all_official_tools() -> None:
    provider, api_key = _require_live_amap_tests()
    tools = _run(load_amap_tools(api_key=api_key, provider=provider))
    tool_names = {tool.name for tool in tools}

    assert set(AMAP_MAPS_TOOLS.tool_names) == tool_names


def test_amap_mcp_all_tools_accept_inputs_and_return_outputs() -> None:
    provider, api_key = _require_live_amap_tests()
    results = _run(_call_all_amap_tools(provider, api_key))

    assert set(AMAP_MAPS_TOOLS.tool_names) == set(results)
    for tool_name, result in results.items():
        _assert_tool_output(tool_name, result)


async def _call_all_amap_tools(provider: str, api_key: str) -> dict[str, Any]:
    tools = {tool.name: tool for tool in await load_amap_tools(api_key=api_key, provider=provider)}
    missing = set(AMAP_MAPS_TOOLS.tool_names) - set(tools)
    assert not missing, f"missing Amap tools: {sorted(missing)}"

    text_search_result = await _invoke_amap_tool(
        tools["maps_text_search"],
        {"keywords": AMAP_TEST_POI, "city": AMAP_TEST_CITY, "citylimit": True},
        "maps_text_search",
    )
    _assert_tool_output("maps_text_search", text_search_result)
    poi_id = _first_poi_id(text_search_result)

    tool_inputs: dict[str, dict[str, Any]] = {
        "maps_weather": {"city": AMAP_TEST_CITY},
        "maps_text_search": {"keywords": AMAP_TEST_POI, "city": AMAP_TEST_CITY, "citylimit": True},
        "maps_search_detail": {"id": poi_id},
        "maps_geo": {"address": f"{AMAP_TEST_CITY}{AMAP_TEST_POI}", "city": AMAP_TEST_CITY},
        "maps_regeocode": {"location": AMAP_TEST_ORIGIN},
        "maps_ip_location": {"ip": "114.114.114.114"},
        "maps_around_search": {
            "keywords": AMAP_TEST_AROUND_KEYWORD,
            "location": AMAP_TEST_ORIGIN,
            "radius": "1000",
            "strategy": 0,
        },
        "maps_distance": {
            "origins": AMAP_TEST_ORIGIN,
            "destination": AMAP_TEST_DESTINATION,
            "type": "1",
        },
        "maps_direction_walking": {
            "origin": AMAP_TEST_ORIGIN,
            "destination": AMAP_TEST_DESTINATION,
        },
        "maps_direction_bicycling": {
            "origin": AMAP_TEST_ORIGIN,
            "destination": AMAP_TEST_DESTINATION,
        },
        "maps_direction_driving": {
            "origin": AMAP_TEST_ORIGIN,
            "destination": AMAP_TEST_DESTINATION,
        },
        "maps_direction_transit_integrated": {
            "origin": AMAP_TEST_ORIGIN,
            "destination": AMAP_TEST_DESTINATION,
            "city": AMAP_TEST_CITY,
            "cityd": AMAP_TEST_CITY,
        },
        "maps_schema_navi": {"lon": "120.155070", "lat": "30.274085"},
        "maps_schema_take_taxi": {
            "slon": "120.155070",
            "slat": "30.274085",
            "sname": AMAP_TEST_POI,
            "dlon": "120.168057",
            "dlat": "30.263363",
            "dname": AMAP_TEST_STATION,
        },
        "maps_schema_personal_map": {
            "orgName": "TravelAgentTest",
            "lineList": [
                {
                    "title": "Day 1",
                    "pointInfoList": [
                        {
                            "name": AMAP_TEST_POI,
                            "lon": 120.155070,
                            "lat": 30.274085,
                            "poiId": poi_id,
                        }
                    ],
                }
            ],
        },
    }

    untested = set(tools) - set(tool_inputs)
    assert not untested, f"add sample inputs for new Amap tools: {sorted(untested)}"

    results: dict[str, Any] = {"maps_text_search": text_search_result}
    for tool_name in AMAP_MAPS_TOOLS.tool_names:
        if tool_name == "maps_text_search":
            continue
        results[tool_name] = await _invoke_amap_tool(tools[tool_name], tool_inputs[tool_name], tool_name)

    return results


def _first_poi_id(text_search_result: Any) -> str:
    pois = _extract_pois(text_search_result)
    for poi in pois:
        poi_id = poi.get("id")
        if poi_id:
            return str(poi_id)
    pytest.fail("maps_text_search did not return a POI id required for detail/schema tests")
