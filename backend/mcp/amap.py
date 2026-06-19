from __future__ import annotations

import asyncio
import json
import math
import os
import re
from datetime import date as Date
from typing import Any

from backend.mcp.client import get_amap_api_key, get_bailian_api_key, load_amap_tools
from backend.schemas.trip import (
    AccommodationAreaResult,
    AttractionResult,
    BudgetLevel,
    LodgingResult,
    McpQuery,
    McpQueryPlan,
    McpResults,
    McpToolName,
    PlaceCategory,
    RouteResult,
    TransportMode,
    WeatherResult,
)


CITY_ALIASES = {
    "hangzhou": "\u676d\u5dde",
    "shanghai": "\u4e0a\u6d77",
    "beijing": "\u5317\u4eac",
    "suzhou": "\u82cf\u5dde",
    "nanjing": "\u5357\u4eac",
    "shanxi": "\u5c71\u897f",
    "taiyuan": "\u592a\u539f",
    "xinzhou": "\u5ffb\u5dde",
    "datong": "\u5927\u540c",
    "jinzhong": "\u664b\u4e2d",
}

PLACE_ALIASES = {
    "west lake": "\u897f\u6e56",
    "hangzhou museum": "\u676d\u5dde\u535a\u7269\u9986",
    "hangzhou art gallery": "\u6d59\u6c5f\u7f8e\u672f\u9986",
    "hangzhou tea house": "\u9752\u85e4\u8336\u9986",
    "wutai mountain": "\u4e94\u53f0\u5c71\u98ce\u666f\u540d\u80dc\u533a",
    "yungang grottoes": "\u4e91\u5188\u77f3\u7a9f",
    "yungang grotto": "\u4e91\u5188\u77f3\u7a9f",
    "pingyao ancient city": "\u5e73\u9065\u53e4\u57ce",
    "jinci temple": "\u664b\u7960\u535a\u7269\u9986",
}


def should_use_amap_mcp(state: dict[str, Any]) -> bool:
    explicit = state.get("mcp_backend")
    if explicit is not None:
        return str(explicit).strip().lower() in {
            "amap",
            "amap_official",
            "official",
            "amap_bailian",
            "bailian",
        }

    env_value = os.getenv("TRAVEL_AGENT_MCP_BACKEND")
    if env_value is not None:
        return env_value.strip().lower() in {"amap", "amap_official", "official", "amap_bailian", "bailian"}

    return bool(get_amap_api_key() or get_bailian_api_key())


def execute_amap_mcp_query_plan(
    query_plan: McpQueryPlan,
    default_city: str,
    provider: str | None = None,
) -> McpResults:
    return asyncio.run(_execute_amap_mcp_query_plan(query_plan, default_city, provider=provider))


async def _execute_amap_mcp_query_plan(
    query_plan: McpQueryPlan,
    default_city: str,
    provider: str | None = None,
) -> McpResults:
    if not query_plan.queries:
        return McpResults()

    tools = {tool.name: tool for tool in await load_amap_tools(provider=provider)}
    collected = McpResults()
    semaphore = asyncio.Semaphore(4)

    async def run_query(query: McpQuery) -> McpResults:
        async with semaphore:
            return await _execute_amap_mcp_query(query=query, default_city=default_city, tools=tools)

    for result in await asyncio.gather(*(run_query(query) for query in query_plan.queries)):
        collected = _merge_results(collected, result)

    return collected


async def _execute_amap_mcp_query(
    query: McpQuery,
    default_city: str,
    tools: dict[str, Any],
) -> McpResults:
    args = query.args
    if query.tool_name == McpToolName.GET_WEATHER:
        city = str(args["city"])
        query_date = _parse_date(str(args["date"]))
        raw = await tools["maps_weather"].ainvoke({"city": _query_city(city)})
        return McpResults(weather=[_parse_weather_result(raw=raw, city=city, query_date=query_date)])

    if query.tool_name == McpToolName.SEARCH_ATTRACTIONS:
        city = str(args["city"])
        raw = await tools["maps_text_search"].ainvoke(
            {
                "keywords": _attraction_keywords(args.get("preferences", [])),
                "city": _query_city(city),
                "citylimit": True,
            }
        )
        return McpResults(attractions=_parse_pois_as_attractions(raw, city=city))

    if query.tool_name == McpToolName.SEARCH_ACCOMMODATION_AREAS:
        city = str(args["city"])
        keyword = (
            "\u4eb2\u5b50 \u9152\u5e97 \u5546\u5708"
            if bool(args.get("prefer_family_room", False))
            else "\u9152\u5e97 \u5546\u5708"
        )
        raw = await tools["maps_text_search"].ainvoke(
            {
                "keywords": keyword,
                "city": _query_city(city),
                "citylimit": True,
            }
        )
        return McpResults(accommodation_areas=_parse_pois_as_accommodation_areas(raw, city=city))

    if query.tool_name == McpToolName.SEARCH_LODGING_NEAR_PLACE:
        city = str(args["city"])
        anchor_place = str(args.get("anchor_place", city))
        radius_km = float(args.get("radius_km", 5))
        anchor_location = await _geocode(tools, anchor_place, city)
        keyword = "\u9152\u5e97 \u6c11\u5bbf"
        raw = None
        if "maps_around_search" in tools:
            raw = await tools["maps_around_search"].ainvoke(
                {
                    "keywords": keyword,
                    "location": anchor_location,
                    "radius": str(int(radius_km * 1000)),
                    "strategy": 0,
                }
            )
        else:
            raw = await tools["maps_text_search"].ainvoke(
                {
                    "keywords": f"{anchor_place} {keyword}",
                    "city": _query_city(city),
                    "citylimit": True,
                }
            )
        return McpResults(
            lodging=_parse_pois_as_lodging_results(
                raw=raw,
                city=city,
                anchor_place=anchor_place,
                anchor_location=anchor_location,
                budget_level=BudgetLevel(str(args.get("budget_level", BudgetLevel.MEDIUM.value))),
            )
        )

    if query.tool_name == McpToolName.GET_ATTRACTION_DETAIL:
        name = str(args["name"])
        city = str(args.get("city", default_city))
        query_date = _parse_date(str(args["date"]))
        search_raw = await tools["maps_text_search"].ainvoke(
            {
                "keywords": _query_place(name, city=city),
                "city": _query_city(city),
                "citylimit": True,
            }
        )
        pois = _extract_pois(search_raw)
        poi = pois[0] if pois else {}
        detail_raw = None
        poi_id = _first_text(poi.get("id"))
        if poi_id and "maps_search_detail" in tools:
            detail_raw = await tools["maps_search_detail"].ainvoke({"id": poi_id})

        return McpResults(
            attractions=[
                _parse_attraction_detail(
                    fallback_name=name,
                    city=city,
                    query_date=query_date,
                    poi=poi,
                    detail_raw=detail_raw or search_raw,
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
        duration, distance = _parse_route_metrics(raw)
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


def _parse_weather_result(raw: Any, city: str, query_date: Date) -> WeatherResult:
    parsed = _maybe_json(raw)
    forecasts = _as_list(parsed.get("forecasts")) if isinstance(parsed, dict) else []
    forecast = next(
        (item for item in forecasts if isinstance(item, dict) and _first_text(item.get("date")) == query_date.isoformat()),
        None,
    )
    if forecast is None and forecasts and isinstance(forecasts[0], dict):
        forecast = forecasts[0]

    if not isinstance(forecast, dict):
        return WeatherResult(city=city, date=query_date, condition="unknown", warning=_compact_raw(raw))

    day_weather = _first_text(forecast.get("dayweather"))
    night_weather = _first_text(forecast.get("nightweather"))
    condition = _normalize_weather_condition(f"{day_weather} {night_weather}")
    warning = (
        f"{forecast.get('date', query_date.isoformat())}: day={day_weather or 'unknown'}, "
        f"night={night_weather or 'unknown'}, temp={forecast.get('nighttemp', '?')}-"
        f"{forecast.get('daytemp', '?')}C"
    )
    return WeatherResult(city=city, date=query_date, condition=condition, warning=warning)


def _parse_pois_as_attractions(raw: Any, city: str) -> list[AttractionResult]:
    results: list[AttractionResult] = []
    for poi in _extract_pois(raw)[:8]:
        name = _first_text(poi.get("name"))
        if not name:
            continue
        results.append(
            AttractionResult(
                name=name,
                city=city,
                category=_category_from_text(
                    " ".join(
                        [
                            _first_text(poi.get("type")),
                            _first_text(poi.get("typecode")),
                            _first_text(poi.get("name")),
                        ]
                    )
                ),
                notes=_compact_raw(poi),
            )
        )
    return results


def _parse_pois_as_accommodation_areas(raw: Any, city: str) -> list[AccommodationAreaResult]:
    results: list[AccommodationAreaResult] = []
    for poi in _extract_pois(raw)[:5]:
        name = _first_text(poi.get("name"))
        if not name:
            continue
        results.append(
            AccommodationAreaResult(
                area_name=name,
                city=city,
                pros=["Amap text-search result", "Useful as a lodging anchor area"],
                cons=[],
                suitable_for=["travelers"],
                estimated_price_level=BudgetLevel.MEDIUM,
                notes=_compact_raw(poi),
            )
        )
    return results


def _parse_pois_as_lodging_results(
    raw: Any,
    city: str,
    anchor_place: str,
    anchor_location: str,
    budget_level: BudgetLevel,
) -> list[LodgingResult]:
    results: list[LodgingResult] = []
    for poi in _extract_pois(raw)[:8]:
        name = _first_text(poi.get("name"))
        if not name:
            continue
        location = _first_text(poi.get("location"))
        results.append(
            LodgingResult(
                name=name,
                city=city,
                area=_first_text(poi.get("adname")) or _first_text(poi.get("business_area")),
                address=_first_text(poi.get("address")),
                location=location,
                anchor_place=anchor_place,
                distance_to_anchor_km=_distance_between_locations_km(anchor_location, location),
                estimated_price_level=budget_level,
                notes=_compact_raw(poi),
            )
        )
    return results


def _parse_attraction_detail(
    fallback_name: str,
    city: str,
    query_date: Date,
    poi: dict[str, Any],
    detail_raw: Any,
) -> AttractionResult:
    detail = _maybe_json(detail_raw)
    if not isinstance(detail, dict):
        detail = {}

    name = _first_text(detail.get("name")) or _first_text(poi.get("name")) or fallback_name
    type_text = " ".join(
        [
            _first_text(detail.get("type")),
            _first_text(detail.get("typecode")),
            _first_text(poi.get("type")),
            _first_text(poi.get("typecode")),
            name,
        ]
    )
    opening_hours = (
        _first_text(detail.get("open_time"))
        or _first_text(detail.get("opentime2"))
        or _first_text(detail.get("opentime"))
        or "Check official venue page"
    )
    return AttractionResult(
        name=name,
        city=city,
        category=_category_from_text(type_text),
        date=query_date,
        is_open=_is_open_from_text(opening_hours),
        opening_hours=opening_hours,
        ticket_price=_parse_price(detail.get("cost")),
        recommended_duration_minutes=120,
        notes=_compact_raw(detail or poi),
    )


async def _geocode(tools: dict[str, Any], address: str, city: str) -> str:
    if _looks_like_location(address):
        return address

    query_address = _query_place(address, city=city)
    query_city = _query_city(city)
    try:
        raw = await tools["maps_geo"].ainvoke({"address": query_address, "city": query_city})
    except Exception:
        location = await _search_location(tools=tools, keyword=query_address, city=query_city)
        if location:
            return location
        raise

    parsed = _maybe_json(raw)
    candidates: list[Any] = []
    if isinstance(parsed, dict):
        candidates.extend(_as_list(parsed.get("geocodes")))
        candidates.extend(_as_list(parsed.get("results")))
    for item in candidates:
        if isinstance(item, dict):
            location = _first_text(item.get("location"))
            if location:
                return location

    match = re.search(r"\d+\.\d+,\d+\.\d+", _raw_text(raw))
    if match:
        return match.group(0)

    raise ValueError(f"Amap geocode did not return a location for {address}")


async def _search_location(tools: dict[str, Any], keyword: str, city: str) -> str:
    if "maps_text_search" not in tools or "maps_search_detail" not in tools:
        return ""
    raw = await tools["maps_text_search"].ainvoke({"keywords": keyword, "city": city, "citylimit": True})
    for poi in _extract_pois(raw):
        poi_id = _first_text(poi.get("id"))
        if not poi_id:
            continue
        detail_raw = await tools["maps_search_detail"].ainvoke({"id": poi_id})
        detail = _maybe_json(detail_raw)
        if isinstance(detail, dict):
            location = _first_text(detail.get("location"))
            if location:
                return location
    return ""


async def _route(tools: dict[str, Any], origin: str, destination: str, mode: TransportMode, city: str):
    if mode == TransportMode.WALK:
        return await tools["maps_direction_walking"].ainvoke({"origin": origin, "destination": destination})
    if mode == TransportMode.TRANSIT:
        return await tools["maps_direction_transit_integrated"].ainvoke(
            {"origin": origin, "destination": destination, "city": _query_city(city), "cityd": _query_city(city)}
        )
    if mode == TransportMode.TAXI:
        return await tools["maps_direction_driving"].ainvoke({"origin": origin, "destination": destination})
    return await tools["maps_distance"].ainvoke({"origins": origin, "destination": destination, "type": "1"})


def _parse_route_metrics(raw: Any) -> tuple[int, float]:
    parsed = _maybe_json(raw)
    duration_seconds, distance_meters = _first_route_metric_pair(parsed)

    if duration_seconds is None:
        duration_seconds = _first_numeric_by_key(parsed, {"duration", "cost", "time"}) or 1800
    if distance_meters is None:
        distance_meters = _first_numeric_by_key(parsed, {"distance"}) or 5000

    duration_minutes = max(1, int(round(float(duration_seconds) / 60)))
    distance_km = round(float(distance_meters) / 1000, 2)
    return duration_minutes, distance_km


def _first_route_metric_pair(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, dict):
        return None, None

    for container_key in ("results", "paths"):
        items = value.get(container_key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return _to_float(items[0].get("duration")), _to_float(items[0].get("distance"))

    route = value.get("route")
    if isinstance(route, dict):
        paths = route.get("paths")
        if isinstance(paths, list) and paths and isinstance(paths[0], dict):
            return _to_float(paths[0].get("duration")), _to_float(paths[0].get("distance"))

    return _to_float(value.get("duration")), _to_float(value.get("distance"))


def _extract_pois(raw: Any) -> list[dict[str, Any]]:
    parsed = _maybe_json(raw)
    if isinstance(parsed, dict):
        pois = parsed.get("pois") or parsed.get("data") or parsed.get("results")
        if isinstance(pois, list):
            return [item for item in pois if isinstance(item, dict)]

    text = _raw_text(raw)
    match = re.search(r'("pois"\s*:\s*\[.*?\])', text, flags=re.S)
    if match:
        parsed = _maybe_json("{" + match.group(1) + "}")
        if isinstance(parsed, dict) and isinstance(parsed.get("pois"), list):
            return [item for item in parsed["pois"] if isinstance(item, dict)]
    return []


def _normalize_weather_condition(text: str) -> str:
    if any(token in text for token in ["\u66b4\u96e8", "\u5927\u96e8", "\u4e2d\u96e8", "\u96f7\u9635\u96e8", "storm"]):
        return "heavy rain"
    if any(token in text for token in ["\u5c0f\u96e8", "\u9635\u96e8", "\u96e8", "rain"]):
        return "rain"
    if any(token in text for token in ["\u96ea", "snow"]):
        return "snow"
    if any(token in text for token in ["\u6674", "sunny", "clear"]):
        return "sunny"
    if any(token in text for token in ["\u9634", "\u4e91", "cloud"]):
        return "cloudy"
    return "unknown"


def _category_from_text(text: str) -> PlaceCategory:
    lowered = text.lower()
    if any(token in lowered for token in ["\u9910", "\u7f8e\u98df", "\u996d\u5e97", "food", "restaurant", "tea"]):
        return PlaceCategory.FOOD
    if any(token in lowered for token in ["\u5546\u573a", "\u8d2d\u7269", "shopping", "mall"]):
        return PlaceCategory.SHOPPING
    if any(token in lowered for token in ["\u535a\u7269\u9986", "\u7f8e\u672f\u9986", "\u5c55\u89c8", "museum", "gallery", "indoor"]):
        return PlaceCategory.INDOOR
    if any(token in lowered for token in ["\u6587\u5316", "\u5386\u53f2", "\u53e4", "\u5bfa", "culture"]):
        return PlaceCategory.CULTURE
    return PlaceCategory.OUTDOOR


def _maybe_json(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1 and _extract_text_from_content_item(value[0]) is not None:
            return _maybe_json(_extract_text_from_content_item(value[0]))
        return value
    if isinstance(value, dict):
        content_text = _extract_text_from_content_item(value)
        if content_text is not None:
            return _maybe_json(content_text)
        return value

    text = _raw_text(value).strip()
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


def _raw_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _extract_text_from_content_item(item)
            parts.append(text if text is not None else str(item))
        return "\n".join(parts)
    text = _extract_text_from_content_item(value)
    if text is not None:
        return text
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_text_from_content_item(value: Any) -> str | None:
    if isinstance(value, dict) and "text" in value:
        return str(value["text"])
    text = getattr(value, "text", None)
    if text is not None:
        return str(text)
    return None


def _first_numeric_by_key(value: Any, keys: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                number = _to_float(item)
                if number is not None:
                    return number
            nested = _first_numeric_by_key(item, keys)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _first_numeric_by_key(item, keys)
            if nested is not None:
                return nested
    return None


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


def _parse_date(value: str | Date) -> Date:
    if isinstance(value, Date):
        return value
    return Date.fromisoformat(value)


def _parse_price(value: Any) -> float:
    text = _first_text(value)
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_open_from_text(value: str) -> bool:
    lowered = value.lower()
    return not any(token in lowered for token in ["closed", "\u6682\u505c", "\u5173\u95ed", "\u4e0d\u5f00\u653e"])


def _looks_like_location(value: str) -> bool:
    return bool(re.fullmatch(r"\s*\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\s*", value))


def _distance_between_locations_km(origin: str, destination: str) -> float:
    origin_pair = _parse_location_pair(origin)
    destination_pair = _parse_location_pair(destination)
    if origin_pair is None or destination_pair is None:
        return 0
    lon1, lat1 = origin_pair
    lon2, lat2 = destination_pair
    # Equirectangular approximation is sufficient for ranking nearby POIs.
    avg_lat = (lat1 + lat2) / 2
    km_per_lat = 111.32
    km_per_lon = 111.32 * max(0.1, abs(math.cos(math.radians(avg_lat))))
    dx = (lon2 - lon1) * km_per_lon
    dy = (lat2 - lat1) * km_per_lat
    return round((dx * dx + dy * dy) ** 0.5, 2)


def _parse_location_pair(value: str) -> tuple[float, float] | None:
    if not _looks_like_location(value):
        return None
    lon_text, lat_text = [part.strip() for part in value.split(",", 1)]
    return float(lon_text), float(lat_text)


def _query_city(city: str) -> str:
    return CITY_ALIASES.get(city.strip().lower(), city)


def _query_place(place: str, city: str = "") -> str:
    lowered = place.strip().lower()
    if lowered in PLACE_ALIASES:
        return PLACE_ALIASES[lowered]

    city_query = _query_city(city) if city else ""
    suffix_map = {
        " museum": "\u535a\u7269\u9986",
        " art gallery": "\u7f8e\u672f\u9986",
        " gallery": "\u7f8e\u672f\u9986",
        " tea house": "\u8336\u9986",
    }
    for suffix, replacement in suffix_map.items():
        if lowered.endswith(suffix) and city_query:
            return f"{city_query}{replacement}"
    return place


def _attraction_keywords(preferences: Any) -> str:
    preference_text = " ".join(str(item) for item in _as_list(preferences) if str(item).strip())
    if preference_text:
        return f"\u666f\u70b9 {preference_text}"
    return "\u666f\u70b9"


def _compact_raw(value: Any) -> str:
    return _raw_text(value)[:500]


def _merge_results(existing: McpResults, incoming: McpResults) -> McpResults:
    weather = {(item.city, item.date): item for item in existing.weather}
    weather.update({(item.city, item.date): item for item in incoming.weather})

    attractions = {(item.name, item.city, item.date): item for item in existing.attractions}
    attractions.update({(item.name, item.city, item.date): item for item in incoming.attractions})

    routes = {(item.origin, item.destination, item.mode): item for item in existing.routes}
    routes.update({(item.origin, item.destination, item.mode): item for item in incoming.routes})

    accommodation_areas = {(item.area_name, item.city): item for item in existing.accommodation_areas}
    accommodation_areas.update({(item.area_name, item.city): item for item in incoming.accommodation_areas})

    lodging = {(item.name, item.city, item.anchor_place): item for item in _mcp_lodging(existing)}
    lodging.update({(item.name, item.city, item.anchor_place): item for item in _mcp_lodging(incoming)})

    return McpResults(
        weather=list(weather.values()),
        attractions=list(attractions.values()),
        routes=list(routes.values()),
        accommodation_areas=list(accommodation_areas.values()),
        lodging=list(lodging.values()),
    )


def _mcp_lodging(mcp_results: McpResults | object) -> list[LodgingResult]:
    value = getattr(mcp_results, "lodging", None)
    if value is None:
        return []
    return value if isinstance(value, list) else []
