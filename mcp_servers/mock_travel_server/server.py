from mcp.server.fastmcp import FastMCP


mcp = FastMCP("mock-travel-server")


RAINY_DAY_SUFFIX = "-01"
CLOSED_DAY_SUFFIX = "-02"
LONG_ROUTE_KEYWORDS = ("West Lake", "Remote", "Mountain")


@mcp.tool()
def get_weather(city: str, date: str):
    """Return deterministic mock weather for a city and date."""
    if date.endswith(RAINY_DAY_SUFFIX):
        return {
            "city": city,
            "date": date,
            "condition": "heavy rain",
            "warning": "Outdoor plans may be affected.",
        }
    return {"city": city, "date": date, "condition": "cloudy", "warning": None}


@mcp.tool()
def get_route_time(origin: str, destination: str, mode: str = "taxi"):
    """Return deterministic mock route duration."""
    long_route = (
        ("West Lake" in origin and "Museum" in destination)
        or ("Museum" in origin and "West Lake" in destination)
        or any(keyword in origin or keyword in destination for keyword in LONG_ROUTE_KEYWORDS[1:])
    )
    return {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "duration_minutes": 150 if long_route else 35,
        "distance_km": 40 if long_route else 8,
    }


@mcp.tool()
def get_attraction_detail(name: str, date: str):
    """Return deterministic mock attraction metadata."""
    closed = name.endswith("Museum") and date.endswith(CLOSED_DAY_SUFFIX)
    return {
        "name": name,
        "category": "indoor" if "Museum" in name or "Gallery" in name else "outdoor",
        "is_open": not closed,
        "opening_hours": "Closed" if closed else "09:00-18:00",
        "ticket_price": 30,
        "recommended_duration_minutes": 120,
        "notes": "Mock closure conflict." if closed else "Mock attraction detail.",
    }


@mcp.tool()
def search_attractions(city: str, preferences: list[str] | None = None):
    """Return deterministic mock attractions by city."""
    preferences = preferences or []
    return [
        {
            "name": "West Lake",
            "city": city,
            "category": "outdoor",
            "match_reason": "Classic outdoor must-see; conflicts with heavy rain on mock rainy days.",
        },
        {
            "name": f"{city} Art Gallery",
            "city": city,
            "category": "indoor",
            "match_reason": "Good rainy-day backup.",
        },
        {
            "name": f"{city} Old Street",
            "city": city,
            "category": "culture",
            "match_reason": f"Matches preferences: {', '.join(preferences) or 'general travel'}.",
        },
    ]


@mcp.tool()
def search_accommodation_areas(city: str, budget_level: str = "medium", prefer_family_room: bool = False):
    """Return deterministic mock accommodation areas instead of specific hotels."""
    family_note = "Family-room friendly." if prefer_family_room else "Standard traveler fit."
    return [
        {
            "area_name": f"{city} Lakeside Area",
            "city": city,
            "pros": ["Close to major sights", "Easy dining access"],
            "cons": ["Can be crowded on holidays"],
            "suitable_for": ["first-time visitors", "relaxed trips", "families"],
            "estimated_price_level": "medium" if budget_level != "low" else "high",
            "notes": family_note,
        },
        {
            "area_name": f"{city} Railway Station Area",
            "city": city,
            "pros": ["Convenient transfers", "Usually better value"],
            "cons": ["Less scenic"],
            "suitable_for": ["budget trips", "early departures"],
            "estimated_price_level": "low" if budget_level == "low" else "medium",
            "notes": family_note,
        },
    ]


@mcp.tool()
def search_lodging_near_place(
    city: str,
    anchor_place: str,
    budget_level: str = "medium",
    prefer_family_room: bool = False,
    radius_km: int = 5,
):
    """Return deterministic mock lodging tied to a concrete attraction anchor."""
    family_note = "Family-room friendly." if prefer_family_room else "Standard traveler fit."
    return [
        {
            "name": f"{anchor_place} Nearby Hotel",
            "city": city,
            "area": f"Near {anchor_place}",
            "address": f"{city} {anchor_place} area",
            "anchor_place": anchor_place,
            "distance_to_anchor_km": 1.2,
            "duration_to_anchor_minutes": 12,
            "estimated_price_level": budget_level,
            "notes": f"{family_note} Within {radius_km} km of the anchor.",
        },
        {
            "name": f"{city} Railway Station Hotel",
            "city": city,
            "area": "Railway station area",
            "address": f"{city} railway station",
            "anchor_place": anchor_place,
            "distance_to_anchor_km": 12,
            "duration_to_anchor_minutes": 45,
            "estimated_price_level": "low",
            "notes": "Convenient for transfers but farther from the anchor.",
        },
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
