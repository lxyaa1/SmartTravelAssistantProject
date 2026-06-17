from __future__ import annotations

import asyncio

from backend.mcp.client import load_mock_tools
from backend.mcp.tool_registry import MOCK_TRAVEL_TOOLS
from mcp_servers.mock_travel_server.server import (
    get_attraction_detail,
    get_route_time,
    get_weather,
    search_accommodation_areas,
    search_attractions,
)


def test_mock_mcp_tools_are_discoverable_over_stdio() -> None:
    tools = asyncio.run(load_mock_tools())
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
