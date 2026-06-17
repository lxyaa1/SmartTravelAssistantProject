# TravelAgent

Local travel assistant skeleton using LangGraph, LangChain, and MCP.

## Current Scope

This is intentionally a minimal runnable skeleton:

- Pydantic schemas for user requests, trip plans, MCP results, and validation issues.
- Structured traveler and accommodation requirements, including children who count as travelers but do not need separate beds.
- LangGraph workflow with a `plan -> collect mock data -> validate -> replan -> final` loop.
- Mock MCP stdio server scaffold.
- No real API calls, login, database, or UI yet.

## Run

```powershell
python -m app.main
```

## Traveler Input

`travelers` can still be a legacy integer, which is treated as the number of adults:

```json
{
  "travelers": 2
}
```

For family trips, prefer the structured form:

```json
{
  "travelers": {
    "adults": 2,
    "children": 1,
    "children_need_bed": 0,
    "children_ages": [6]
  }
}
```

In this example, `total_people` is 3 for tickets, transport, food, and route intensity. `bed_count` is 2 for accommodation planning.

## Workflow

```text
parse_request
  -> initial_plan
  -> collect_mcp_data
  -> validate_plan
  -> replan or final_writer
```

The mock data intentionally creates bad weather and long route issues so the replanning branch can be exercised.

## Mock MCP Server

The local mock MCP server runs over stdio:

```powershell
python mcp_servers/mock_travel_server/server.py
```

Available tools:

```text
get_weather(city, date)
get_route_time(origin, destination, mode)
get_attraction_detail(name, date)
search_attractions(city, preferences)
```

The mock data intentionally includes conflicts:

- Dates ending in `-01` return heavy rain, making outdoor plans risky.
- Museum visits on dates ending in `-02` return closed.
- Routes involving `West Lake`, `Remote`, or `Mountain` return a long 150-minute transfer.
- `search_attractions` returns both outdoor options and indoor rainy-day backups.

## Next Steps

1. Replace deterministic node logic with LangChain LLM calls that still return the same Pydantic schemas.
2. Move `collect_mcp_data_node` from in-process mock data to `langchain-mcp-adapters`.
3. Add a Streamlit page after the backend loop is stable.
