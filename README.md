# TravelAgent

Local travel assistant skeleton using LangGraph, LangChain, and MCP.

## Current Scope

This is intentionally a minimal runnable skeleton:

- Pydantic schemas for user requests, trip plans, MCP results, and validation issues.
- Structured traveler and accommodation requirements, including children who count as travelers but do not need separate beds.
- LangGraph workflow with pre-plan MCP queries, plan-check MCP queries, validation, and replanning.
- Mock MCP stdio server scaffold plus optional Amap Maps MCP backend.
- Local Streamlit UI with no login, database, or registration.

## Run

```powershell
python -m app.main
```

For the local UI:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8502
```

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

- `initial_plan`: generates a structured `TripPlan`.
- `replan`: revises the current structured `TripPlan` based on validation issues.

The query planners, MCP data collection, validation routing, and final markdown rendering remain deterministic Python nodes.

In the Streamlit UI, the `LLM` checkbox only controls the two planning nodes:

```text
initial_plan
replan
```

When unchecked, these nodes use deterministic Python logic. When checked, they call DashScope/Qwen and still must return the same `TripPlan` Pydantic schema.

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
get_attraction_detail       -> maps_text_search + maps_search_detail
get_route_time              -> maps_geo + direction/distance tools
```

If Amap MCP fails, the workflow records the error in `mcp_errors` and falls back to the local mock MCP data so the graph can still complete. To force mock MCP:

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
  -> preplan_query_planner
  -> collect_preplan_mcp_data
  -> initial_plan
  -> plan_check_query_planner
  -> collect_plan_mcp_data
  -> validate_plan
  -> replan or final_writer
```

If validation finds `high` or `critical` issues and the loop has not reached `max_iterations`, `replan` sends the flow back to `plan_check_query_planner`.

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
```

The mock data intentionally includes conflicts:

- Dates ending in `-01` return heavy rain, making outdoor plans risky.
- Museum visits on dates ending in `-02` return closed.
- Routes between `West Lake` and a museum, or routes involving `Remote` / `Mountain`, return a long 150-minute transfer.
- `search_attractions` returns both outdoor options and indoor rainy-day backups.
- `search_accommodation_areas` returns area-level lodging options, not specific hotels.

## Next Steps

1. Replace deterministic node logic with LangChain LLM calls that still return the same Pydantic schemas.
2. Replace remaining mock-only fields with richer Amap MCP result parsing.
3. Add a Streamlit page after the backend loop is stable.
