from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date as Date
from typing import Any

from backend.mcp.client import load_amap_tools
from backend.schemas.trip import (
    AccommodationAreaResult,
    AttractionResult,
    BudgetLevel,
    McpQuery,
    McpQueryPlan,
    McpResults,
    McpToolName,
    PlaceCategory,
    RouteResult,
    TransportMode,
    WeatherResult,
)


def should_use_amap_mcp(state: dict[str, Any]) -> bool:
    explicit = state.get("mcp_backend")
    if explicit is not None:
        return str(explicit).lower() == "amap"

    env_value = os.getenv("TRAVEL_AGENT_MCP_BACKEND")
    if env_value is not None:
        return env_value.strip().lower() == "amap"

    return bool(os.getenv("DASHSCOPE_API_KEY"))


def execute_amap_mcp_query_plan(
    query_plan: McpQueryPlan,
    default_city: str,
) -> McpResults:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is required when Amap MCP backend is enabled")

    return asyncio.run(_execute_amap_mcp_query_plan(query_plan, default_city, api_key))


async def _execute_amap_mcp_query_plan(
    query_plan: McpQueryPlan,
    default_city: str,
    api_key: str,
) -> McpResults:
    tools = {tool.name: tool for tool in await load_amap_tools(api_key)}
    collected = McpResults()

    for query in query_plan.queries:
        result = await _execute_amap_mcp_query(query=query, default_city=default_city, tools=tools)
        collected = _merge_results(collected, result)

    return collected


async def _execute_amap_mcp_query(
    query: McpQuery,
    default_city: str,
    tools: dict[str, Any],
) -> McpResults:
    args = query.args
    if query.tool_name == McpToolName.GET_WEATHER:
        raw = await tools["maps_weather"].ainvoke({"city": str(args["city"])})
        return McpResults(
            weather=[
                WeatherResult(
                    city=str(args["city"]),
                    date=_parse_date(str(args["date"])),
                    condition=_weather_condition(raw),
                    warning=_compact_raw(raw),
                )
            ]
        )

    if query.tool_name == McpToolName.SEARCH_ATTRACTIONS:
        city = str(args["city"])
        preferences = " ".join(str(item) for item in args.get("preferences", []))
        raw = await tools["maps_text_search"].ainvoke(
            {
                "keywords": f"景点 {preferences}".strip(),
                "city": city,
                "citylimit": True,
            }
        )
        return McpResults(attractions=_parse_pois_as_attractions(raw, city=city))

    if query.tool_name == McpToolName.SEARCH_ACCOMMODATION_AREAS:
        city = str(args["city"])
        keyword = "亲子 酒店 商圈" if bool(args.get("prefer_family_room", False)) else "酒店 商圈"
        raw = await tools["maps_text_search"].ainvoke(
            {
                "keywords": keyword,
                "city": city,
                "citylimit": True,
            }
        )
        return McpResults(accommodation_areas=_parse_pois_as_accommodation_areas(raw, city=city))

    if query.tool_name == McpToolName.GET_ATTRACTION_DETAIL:
        name = str(args["name"])
        query_date = _parse_date(str(args["date"]))
        raw = await tools["maps_text_search"].ainvoke(
            {
                "keywords": name,
                "city": default_city,
                "citylimit": True,
            }
        )
        pois = _extract_pois(raw)
        poi = pois[0] if pois else {}
        detail_raw = None
        poi_id = _first_text(poi.get("id"))
        if poi_id and "maps_search_detail" in tools:
            detail_raw = await tools["maps_search_detail"].ainvoke({"id": poi_id})

        return McpResults(
            attractions=[
                AttractionResult(
                    name=_first_text(poi.get("name")) or name,
                    city=_first_text(poi.get("cityname")) or default_city,
                    category=_category_from_text(_first_text(poi.get("type")) or name),
                    date=query_date,
                    is_open=True,
                    opening_hours="Check official venue page",
                    ticket_price=0,
                    recommended_duration_minutes=120,
                    notes=_compact_raw(detail_raw or raw),
                )
            ]
        )

    if query.tool_name == McpToolName.GET_ROUTE_TIME:
        origin_name = str(args["origin"])
        destination_name = str(args["destination"])
        mode = TransportMode(str(args.get("mode", TransportMode.TAXI.value)))
        origin_location = await _geocode(tools, origin_name, default_city)
        destination_location = await _geocode(tools, destination_name, default_city)
        raw = await _route(tools, origin_location, destination_location, mode, default_city)
        duration, distance = _parse_route_metrics(raw, mode=mode)
        return McpResults(
            routes=[
                RouteResult(
                    origin=origin_name,
                    destination=destination_name,
                    mode=mode,
                    duration_minutes=duration,
                    distance_km=distance,
                )
            ]
        )

    raise ValueError(f"Unsupported Amap MCP query: {query.tool_name}")


async def _geocode(tools: dict[str, Any], address: str, city: str) -> str:
    raw = await tools["maps_geo"].ainvoke({"address": address, "city": city})
    parsed = _maybe_json(raw)
    geocodes = _as_list(parsed.get("geocodes")) if isinstance(parsed, dict) else []
    if geocodes:
        location = _first_text(geocodes[0].get("location"))
        if location:
            return location

    match = re.search(r"\d+\.\d+,\d+\.\d+", str(raw))
    if match:
        return match.group(0)

    raise ValueError(f"Amap geocode did not return a location for {address}")


async def _route(tools: dict[str, Any], origin: str, destination: str, mode: TransportMode, city: str):
    if mode == TransportMode.WALK:
        return await tools["maps_direction_walking"].ainvoke({"origin": origin, "destination": destination})
    if mode == TransportMode.TRANSIT:
        return await tools["maps_direction_transit_integrated"].ainvoke(
            {"origin": origin, "destination": destination, "city": city, "cityd": city}
        )
    if mode == TransportMode.TAXI:
        return await tools["maps_direction_driving"].ainvoke({"origin": origin, "destination": destination})
    return await tools["maps_distance"].ainvoke({"origins": origin, "destination": destination, "type": "1"})


def _parse_route_metrics(raw: Any, mode: TransportMode) -> tuple[int, float]:
    parsed = _maybe_json(raw)
    text = str(raw)

    durations = _find_numeric_values(parsed, keys={"duration", "cost", "time"})
    distances = _find_numeric_values(parsed, keys={"distance"})

    if not durations:
        durations = [float(item) for item in re.findall(r'"duration"\s*:\s*"?(\d+(?:\.\d+)?)"?', text)]
    if not distances:
        distances = [float(item) for item in re.findall(r'"distance"\s*:\s*"?(\d+(?:\.\d+)?)"?', text)]

    duration_seconds = durations[0] if durations else 1800
    distance_meters = distances[0] if distances else 5000

    if mode == TransportMode.TRANSIT and duration_seconds < 300:
        duration_minutes = int(duration_seconds)
    else:
        duration_minutes = max(1, int(round(duration_seconds / 60)))

    distance_km = round(distance_meters / 1000, 2) if distance_meters > 100 else round(distance_meters, 2)
    return duration_minutes, distance_km


def _parse_pois_as_attractions(raw: Any, city: str) -> list[AttractionResult]:
    pois = _extract_pois(raw)
    results: list[AttractionResult] = []
    for poi in pois[:8]:
        name = _first_text(poi.get("name"))
        if not name:
            continue
        type_text = _first_text(poi.get("type")) or ""
        results.append(
            AttractionResult(
                name=name,
                city=_first_text(poi.get("cityname")) or city,
                category=_category_from_text(type_text),
                notes=_compact_raw(poi),
            )
        )
    return results


def _parse_pois_as_accommodation_areas(raw: Any, city: str) -> list[AccommodationAreaResult]:
    pois = _extract_pois(raw)
    results: list[AccommodationAreaResult] = []
    for poi in pois[:5]:
        name = _first_text(poi.get("name"))
        if not name:
            continue
        results.append(
            AccommodationAreaResult(
                area_name=name,
                city=_first_text(poi.get("cityname")) or city,
                pros=["Amap text-search result", "Useful as a lodging anchor area"],
                cons=[],
                suitable_for=["travelers"],
                estimated_price_level=BudgetLevel.MEDIUM,
                notes=_compact_raw(poi),
            )
        )
    return results


def _extract_pois(raw: Any) -> list[dict[str, Any]]:
    parsed = _maybe_json(raw)
    if isinstance(parsed, dict):
        pois = parsed.get("pois") or parsed.get("data") or parsed.get("results")
        if isinstance(pois, list):
            return [item for item in pois if isinstance(item, dict)]

    text = str(raw)
    match = re.search(r'("pois"\s*:\s*\[.*?\])', text, flags=re.S)
    if match:
        parsed = _maybe_json("{" + match.group(1) + "}")
        if isinstance(parsed, dict) and isinstance(parsed.get("pois"), list):
            return [item for item in parsed["pois"] if isinstance(item, dict)]
    return []


def _weather_condition(raw: Any) -> str:
    text = str(raw).lower()
    if any(token in text for token in ["雨", "rain", "storm"]):
        return "rain"
    if any(token in text for token in ["雪", "snow"]):
        return "snow"
    if any(token in text for token in ["晴", "sunny", "clear"]):
        return "sunny"
    if any(token in text for token in ["阴", "云", "cloud"]):
        return "cloudy"
    return "unknown"


def _category_from_text(text: str) -> PlaceCategory:
    lowered = text.lower()
    if any(token in lowered for token in ["餐", "美食", "food", "restaurant", "tea"]):
        return PlaceCategory.FOOD
    if any(token in lowered for token in ["商场", "购物", "shopping", "mall"]):
        return PlaceCategory.SHOPPING
    if any(token in lowered for token in ["博物馆", "馆", "museum", "gallery", "室内"]):
        return PlaceCategory.INDOOR
    if any(token in lowered for token in ["文化", "历史", "culture"]):
        return PlaceCategory.CULTURE
    return PlaceCategory.OUTDOOR


def _maybe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return value
    return value


def _find_numeric_values(value: Any, keys: set[str]) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    pass
            values.extend(_find_numeric_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            values.extend(_find_numeric_values(item, keys))
    return values


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def _compact_raw(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return text[:500]


def _merge_results(existing: McpResults, incoming: McpResults) -> McpResults:
    weather = {(item.city, item.date): item for item in existing.weather}
    weather.update({(item.city, item.date): item for item in incoming.weather})

    attractions = {(item.name, item.city, item.date): item for item in existing.attractions}
    attractions.update({(item.name, item.city, item.date): item for item in incoming.attractions})

    routes = {(item.origin, item.destination, item.mode): item for item in existing.routes}
    routes.update({(item.origin, item.destination, item.mode): item for item in incoming.routes})

    accommodation_areas = {(item.area_name, item.city): item for item in existing.accommodation_areas}
    accommodation_areas.update({(item.area_name, item.city): item for item in incoming.accommodation_areas})

    return McpResults(
        weather=list(weather.values()),
        attractions=list(attractions.values()),
        routes=list(routes.values()),
        accommodation_areas=list(accommodation_areas.values()),
    )
