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


AMAP_MAPS_TOOLS = ToolGroup(
    name="amap_maps",
    tool_names=(
        "maps_weather",
        "maps_text_search",
        "maps_search_detail",
        "maps_geo",
        "maps_direction_driving",
        "maps_direction_walking",
        "maps_direction_transit_integrated",
        "maps_distance",
    ),
)
