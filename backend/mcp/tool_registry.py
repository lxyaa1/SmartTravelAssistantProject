from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolGroup:
    name: str
    tool_names: tuple[str, ...]


MOCK_TRAVEL_TOOLS = ToolGroup(
    name="mock_travel",
    tool_names=(
        "get_weather",
        "get_route_time",
        "get_attraction_detail",
        "search_attractions",
        "search_accommodation_areas",
    ),
)
