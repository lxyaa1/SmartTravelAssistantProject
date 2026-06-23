# TravelAgent

Local travel assistant skeleton using LangGraph, LangChain, and MCP.

## Current Scope

This is a local runnable prototype:

- Pydantic schemas for user requests, trip plans, MCP results, and validation issues.
- Structured traveler and accommodation requirements, including children who count as travelers but do not need separate beds.
- Structured route skeletons, route segments, accommodation stays, daily move/stay timelines, and plan quality gates.
- LangGraph workflow with city-route planning, pre-plan MCP queries, day schedule drafting, plan-check MCP queries, validation, repair strategy selection, and replanning.
- Mock MCP stdio server scaffold plus optional Amap Maps MCP backend.
- Local Streamlit UI with workflow streaming, log visualization, no login, database, or registration.

## Run

```powershell
python -m app.main
```

For the local UI:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8502
```

For a static multi-case planning report:

```powershell
python scripts/batch_plan_report.py --backend mock --max-iterations 1
```

The script writes an HTML report to `data/reports/`. To run only the Shanxi case with live Amap MCP:

```powershell
python scripts/batch_plan_report.py --backend amap --case Shanxi --max-iterations 1
```

Add `--use-llm` when you want the LLM-backed planning nodes to run, and `--open` to open the generated HTML file in the default browser.

## LLM Configuration

The project can use Alibaba Cloud Model Studio / DashScope through its OpenAI-compatible chat completions API.

Required environment variable:

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

Optional environment variables:

```powershell
$env:DASHSCOPE_MODEL="qwen-plus"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:TRAVEL_AGENT_USE_LLM="true"
```

If `DASHSCOPE_API_KEY` is present, LLM mode is enabled by default. To force deterministic mock planning:

```powershell
$env:TRAVEL_AGENT_USE_LLM="false"
```

Current LLM-backed nodes:

- `city_route_planner`: can generate a structured `CityRoutePlan`.
- `draft_day_schedule`: generates a structured `TripPlan`.
- `replan`: revises the current structured `TripPlan` based on validation issues.

The query planners, MCP data collection, validation routing, and final markdown rendering remain deterministic Python nodes.

In the Streamlit UI, the `LLM` checkbox controls the planning/replanning nodes:

```text
city_route_planner
draft_day_schedule
replan
```

When unchecked, these nodes use deterministic Python logic. When checked, they call DashScope/Qwen and still must return the same `TripPlan` Pydantic schema.

## Trip Plan Schema

The daily itinerary is timeline-only. `PlanDay` no longer has separate `visits`, `schedule_blocks`, or daily transfer fields.

Each `PlanDay.timeline` item is one primitive:

```text
stay: at one place for one purpose, such as sleep, meal, visit, rest, check-in, or checkout
move: from one concrete point to another, with mode, purpose, duration, distance, cost, and notes
```

This keeps the internal plan consistent: a trip is either staying somewhere or moving from one point to another.

The Streamlit UI streams workflow node updates while a plan is running and writes action logs to:

```text
data/logs/*.jsonl
```

## MCP Backend Configuration

The recommended live map backend is Amap's official MCP endpoint. It uses an Amap Web service key:

```powershell
$env:Amap_Key="your-amap-web-service-key"
$env:TRAVEL_AGENT_MCP_BACKEND="amap"
$env:TRAVEL_AGENT_AMAP_PROVIDER="official"
```

The official Amap MCP endpoint is:

```text
https://mcp.amap.com/mcp?key=...
```

The project can also call Bailian's Amap Maps MCP endpoint through Streamable HTTP. It uses the same `DASHSCOPE_API_KEY` environment variable.

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
$env:TRAVEL_AGENT_MCP_BACKEND="amap"
$env:TRAVEL_AGENT_AMAP_PROVIDER="bailian"
```

The Bailian Amap MCP endpoint is:

```text
https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp
```

Mapped Amap tools:

```text
maps_weather
maps_text_search
maps_search_detail
maps_geo
maps_direction_driving
maps_direction_walking
maps_direction_transit_integrated
maps_distance
```

Internal query mapping:

```text
get_weather                 -> maps_weather
search_attractions          -> maps_text_search
search_accommodation_areas  -> maps_text_search
search_lodging_near_place   -> maps_geo + maps_around_search/maps_text_search
get_attraction_detail       -> maps_text_search + maps_search_detail
get_route_time              -> maps_geo + direction/distance tools
```

If Amap MCP fails, the workflow records the error in `mcp_errors` and does not mix local mock data into the live result. Missing live data is treated as unverified MCP data during validation. To force mock MCP for local development:

```powershell
$env:TRAVEL_AGENT_MCP_BACKEND="mock"
```

For a minimal connectivity check:

```powershell
$env:Amap_Key="your-amap-web-service-key"
python scripts/check_amap_mcp.py --provider official
python scripts/check_amap_mcp.py --provider official --call-weather --city 杭州
```

The official Amap MCP endpoint is `https://mcp.amap.com/mcp?key=...`. The first command only lists tools. The second command also calls `maps_weather`, which helps separate connection problems from Amap tool authorization problems.

To keep testing Bailian's Amap MCP endpoint:

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
python scripts/check_amap_mcp.py --provider bailian --call-weather --city 杭州
```

For a reference-style connectivity check matching `travel-agent-main/mcp_client.py`:

```powershell
python scripts/check_amap_mcp_like_reference.py
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
  -> city_route_planner
  -> preplan_query_planner
  -> collect_preplan_mcp_data
  -> draft_day_schedule
  -> plan_check_query_planner
  -> collect_plan_mcp_data
  -> validate_plan
  -> repair_strategy_planner
  -> replan, final_writer, or infeasible final output
```

If validation finds `high` or `critical` issues and the loop has not reached `max_iterations`, `repair_strategy_planner` sends the flow to `replan`. If the maximum loop count is reached, the final output is marked provisional through `TripPlan.quality_gate`.

```text
replan
  -> plan_check_query_planner
  -> collect_plan_mcp_data
  -> validate_plan
```

The MCP data collection node consumes `pending_mcp_queries`, merges results into `mcp_results`, and clears the pending query list.

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
search_accommodation_areas(city, budget_level, prefer_family_room)
search_lodging_near_place(city, anchor_place, budget_level, prefer_family_room, radius_km)
```

The mock data intentionally includes conflicts:

- Dates ending in `-01` return heavy rain, making outdoor plans risky.
- Museum visits on dates ending in `-02` return closed.
- Routes between `West Lake` and a museum, or routes involving `Remote` / `Mountain`, return a long 150-minute transfer.
- `search_attractions` returns both outdoor options and indoor rainy-day backups.
- `search_accommodation_areas` returns area-level fallback lodging options.
- `search_lodging_near_place` returns concrete mock lodging tied to an attraction anchor, including a nearby option and a farther station-area fallback.

## Next Steps

1. Add a dedicated train/flight/ticket source; Amap MCP cannot provide reliable train numbers or ticket inventory.
2. Improve city and region inference beyond the current deterministic rules for common Shanxi anchors.
3. Add stronger LLM prompts and evaluation cases for multi-city itinerary repairs.
