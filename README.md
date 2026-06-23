# TravelAgent

基于 LangGraph、LangChain 和 MCP 的本地旅行助手原型。

## 当前范围

这是一个可在本地运行的原型项目：

- 使用 Pydantic schema 描述用户请求、旅行计划、MCP 结果和校验问题。
- 支持结构化的旅客和住宿需求，包括儿童计入出行人数但不一定需要单独占床。
- 支持结构化路线骨架、路线段、住宿安排、每日 move/stay 时间线和计划质量门禁。
- 使用 LangGraph 串联城市路线规划、预规划 MCP 查询、每日行程草稿、计划校验 MCP 查询、校验、修复策略选择和重规划。
- 提供 mock MCP stdio server，并可选接入高德地图 MCP 后端。
- 提供本地 Streamlit UI，支持工作流流式展示和日志可视化；不包含登录、数据库或注册能力。

## 运行

```powershell
python -m app.main
```

运行本地 UI：

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8502
```

生成静态多案例规划报告：

```powershell
python scripts/batch_plan_report.py --backend mock --max-iterations 1
```

脚本会将 HTML 报告写入 `data/reports/`。如果只想用真实高德 MCP 跑山西案例：

```powershell
python scripts/batch_plan_report.py --backend amap --case Shanxi --max-iterations 1
```

需要运行 LLM 节点时添加 `--use-llm`；需要在默认浏览器打开生成的 HTML 文件时添加 `--open`。

## LLM 配置

项目可以通过 OpenAI 兼容的 chat completions API 调用阿里云百炼 / DashScope。

必需环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

可选环境变量：

```powershell
$env:DASHSCOPE_MODEL="qwen-plus"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:TRAVEL_AGENT_USE_LLM="true"
```

如果存在 `DASHSCOPE_API_KEY`，默认会启用 LLM 模式。强制使用 deterministic mock planning：

```powershell
$env:TRAVEL_AGENT_USE_LLM="false"
```

当前会调用 LLM 的节点：

- `city_route_planner`：生成结构化 `CityRoutePlan`。
- `draft_day_schedule`：生成结构化 `TripPlan`。
- `replan`：根据校验问题修改当前结构化 `TripPlan`。

查询规划、MCP 数据采集、校验路由和最终 markdown 渲染仍然是 deterministic Python 节点。

在 Streamlit UI 中，`LLM` 复选框控制以下规划 / 重规划节点：

```text
city_route_planner
draft_day_schedule
replan
```

未勾选时，这些节点使用 deterministic Python 逻辑。勾选后，它们会调用 DashScope/Qwen，但仍必须返回相同的 `TripPlan` Pydantic schema。

## Trip Plan Schema

每日行程只使用 timeline 表示。`PlanDay` 不再单独包含 `visits`、`schedule_blocks` 或每日交通字段。

每个 `PlanDay.timeline` item 都是一个原子动作：

```text
stay: 在某个地点停留并执行一个目的，例如 sleep、meal、visit、rest、check-in 或 checkout
move: 从一个具体点移动到另一个具体点，包含交通方式、目的、耗时、距离、费用和备注
```

这样可以保持内部计划一致：旅行过程要么是在某地停留，要么是从一个点移动到另一个点。

Streamlit UI 会在计划运行时流式展示工作流节点更新，并将行动日志写入：

```text
data/logs/*.jsonl
```

## MCP 后端配置

推荐的真实地图后端是高德官方 MCP endpoint，需要高德 Web 服务 key：

```powershell
$env:Amap_Key="your-amap-web-service-key"
$env:TRAVEL_AGENT_MCP_BACKEND="amap"
$env:TRAVEL_AGENT_AMAP_PROVIDER="official"
```

高德官方 MCP endpoint：

```text
https://mcp.amap.com/mcp?key=...
```

项目也可以通过 Streamable HTTP 调用百炼 Amap Maps MCP endpoint，使用同一个 `DASHSCOPE_API_KEY` 环境变量。

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
$env:TRAVEL_AGENT_MCP_BACKEND="amap"
$env:TRAVEL_AGENT_AMAP_PROVIDER="bailian"
```

百炼 Amap MCP endpoint：

```text
https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp
```

已映射的高德工具：

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

内部查询映射：

```text
get_weather                 -> maps_weather
search_attractions          -> maps_text_search
search_accommodation_areas  -> maps_text_search
search_lodging_near_place   -> maps_geo + maps_around_search/maps_text_search
get_attraction_detail       -> maps_text_search + maps_search_detail
get_route_time              -> maps_geo + direction/distance tools
```

如果 Amap MCP 失败，工作流会将错误记录到 `mcp_errors`，不会把本地 mock 数据混入真实结果。缺失的真实数据会在校验阶段被视为未验证 MCP 数据。如果要在本地开发中强制使用 mock MCP：

```powershell
$env:TRAVEL_AGENT_MCP_BACKEND="mock"
```

最小连通性检查：

```powershell
$env:Amap_Key="your-amap-web-service-key"
python scripts/check_amap_mcp.py --provider official
python scripts/check_amap_mcp.py --provider official --call-weather --city 杭州
```

第一个命令只列出工具。第二个命令还会调用 `maps_weather`，用于区分连接问题和高德工具授权问题。

继续测试百炼 Amap MCP endpoint：

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
python scripts/check_amap_mcp.py --provider bailian --call-weather --city 杭州
```

运行与 `travel-agent-main/mcp_client.py` 类似的参考连通性检查：

```powershell
python scripts/check_amap_mcp_like_reference.py
```

## Traveler Input

`travelers` 仍可使用旧版整数形式，此时会被视为成人数量：

```json
{
  "travelers": 2
}
```

家庭出行建议使用结构化形式：

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

在这个例子中，`total_people` 是 3，用于门票、交通、餐饮和路线强度计算；`bed_count` 是 2，用于住宿规划。

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

如果校验发现 `high` 或 `critical` 问题，并且还没有达到 `max_iterations`，`repair_strategy_planner` 会将流程送到 `replan`。如果达到最大循环次数，最终输出会通过 `TripPlan.quality_gate` 标记为 provisional。

```text
replan
  -> plan_check_query_planner
  -> collect_plan_mcp_data
  -> validate_plan
```

MCP 数据采集节点会消费 `pending_mcp_queries`，将结果合并到 `mcp_results`，然后清空 pending query list。

## Mock MCP Server

本地 mock MCP server 使用 stdio 运行：

```powershell
python mcp_servers/mock_travel_server/server.py
```

可用工具：

```text
get_weather(city, date)
get_route_time(origin, destination, mode)
get_attraction_detail(name, date)
search_attractions(city, preferences)
search_accommodation_areas(city, budget_level, prefer_family_room)
search_lodging_near_place(city, anchor_place, budget_level, prefer_family_room, radius_km)
```

mock 数据有意包含冲突：

- 日期以 `-01` 结尾时返回 heavy rain，使户外计划存在风险。
- 博物馆类访问在日期以 `-02` 结尾时返回 closed。
- `West Lake` 与 museum 之间的路线，或包含 `Remote` / `Mountain` 的路线，会返回 150 分钟的长距离交通。
- `search_attractions` 同时返回户外选项和室内雨天备选。
- `search_accommodation_areas` 返回区域级住宿备选。
- `search_lodging_near_place` 返回与景点 anchor 绑定的具体 mock 住宿，包括附近选项和更远的车站区域备选。

## Next Steps

1. 添加专门的火车 / 航班 / 票务数据源；Amap MCP 不能可靠提供车次、航班号或余票。
2. 改进城市和区域推断，超出当前对常见山西 anchor 的 deterministic 规则。
3. 增强 LLM prompt，并增加多城市行程修复的评测案例。
