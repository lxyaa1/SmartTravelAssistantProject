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
        "search_lodging_near_place",
    ),
)


AMAP_MAPS_TOOLS = ToolGroup(
    name="amap_maps",
    tool_names=(
        "maps_direction_bicycling",
        "maps_weather",
        "maps_text_search",
        "maps_search_detail",
        "maps_geo",
        "maps_regeocode",
        "maps_ip_location",
        "maps_schema_personal_map",
        "maps_around_search",
        "maps_direction_driving",
        "maps_direction_walking",
        "maps_direction_transit_integrated",
        "maps_distance",
        "maps_schema_navi",
        "maps_schema_take_taxi",
    ),
)
