from __future__ import annotations

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
    long_route = any(keyword in origin or keyword in destination for keyword in LONG_ROUTE_KEYWORDS)
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
