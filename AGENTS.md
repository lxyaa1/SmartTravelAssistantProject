# TravelAgent 项目协作说明

本文档记录当前项目的重要事实、工程约定、运行方式、调试入口，以及基于最新日志看到的主要问题。后续 Agent 接手项目时，优先阅读本文件，再进入具体代码。

## 项目定位

- 这是一个本地运行的旅游助手原型项目，不包含用户注册、登录、数据库、多租户等线上产品能力。
- 核心目标是：用户输入出发地、目的地、日期、人数、预算、偏好、必去景点、避开项后，系统生成结构化旅行计划，并通过 MCP 地图/天气/景点/住宿数据反复校验和重规划。
- 技术栈：Python 3.11+、Pydantic v2、LangGraph、LangChain MCP adapters、OpenAI-compatible DashScope/Qwen、Streamlit。
- 当前项目仍处在个人开发阶段，可以为了正确性调整 schema，不必强行兼容旧演示数据。

## 关键目录

- `app/main.py`：最小 CLI/demo 入口。
- `app/streamlit_app.py`：本地 Streamlit 前端，支持运行 workflow、实时展示当前计划/问题/MCP 数据、查看 JSONL 日志。
- `backend/schemas/trip.py`：项目最重要的“合同层”，所有 Agent/节点输入输出都应走 Pydantic schema，内部不应传自由文本。
- `backend/graph/workflow.py`：LangGraph 工作流定义。
- `backend/graph/state.py`：全局 state 字段定义。
- `backend/graph/routing.py`：校验后的路由判断。
- `backend/agents/nodes.py`：各节点实现，包含 deterministic 规划、MCP 查询规划、校验、修复、mock MCP 调用等。
- `backend/agents/llm.py`：DashScope/Qwen 调用与结构化 JSON 修复。
- `backend/mcp/client.py`：MCP client 构造，包含 mock stdio、高德官方 MCP、百炼 Amap MCP。
- `backend/mcp/amap.py`：内部 MCP 查询到高德 MCP 工具的映射、解析和归一化。
- `backend/mcp/tool_registry.py`：mock 与 Amap 可用工具清单。
- `mcp_servers/mock_travel_server/server.py`：本地 mock MCP stdio server。
- `tests/`：schema、workflow、LLM JSON 修复、MCP 解析、高德异常路线等测试。
- `data/logs/*.jsonl`：Streamlit 每次运行生成的行动日志。

## 运行方式

CLI/demo：

```powershell
python -m app.main
```

Streamlit 前端：

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8502
```

本地测试：

```powershell
python -m pytest -q
```

高德路线异常 live test 需要显式打开，默认会 skip：

```powershell
$env:RUN_AMAP_MCP_LIVE_TESTS="1"
$env:TRAVEL_AGENT_AMAP_PROVIDER="official"
python -m pytest tests/test_amap_route_anomalies.py -q -s
```

## 环境变量

LLM：

- `DASHSCOPE_API_KEY`：DashScope/Qwen API key。
- `DASHSCOPE_MODEL`：默认 `qwen-plus`。
- `DASHSCOPE_BASE_URL`：默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- `TRAVEL_AGENT_USE_LLM=false`：即使有 key 也禁用 LLM，改用 deterministic 逻辑。

MCP：

- `TRAVEL_AGENT_MCP_BACKEND=mock|amap`：选择 mock 或高德 MCP。
- `TRAVEL_AGENT_AMAP_PROVIDER=auto|official|bailian`：高德来源，默认逻辑优先 official。
- `Amap_Key` / `AMAP_KEY` / `AMAP_MAPS_API_KEY`：高德官方 MCP key。
- `DASHSCOPE_API_KEY`：百炼 Amap MCP 使用的 key。
- `TRAVEL_AGENT_AMAP_CONCURRENCY`：高德 MCP 并发数，当前默认 `1`，降低 QPS 风险。
- `TRAVEL_AGENT_AMAP_REQUEST_INTERVAL_SECONDS`：高德请求间隔，当前默认 `0.2` 秒。
- `TRAVEL_AGENT_ALLOW_MOCK_MCP_FALLBACK=true`：显式允许 Amap 失败后 fallback 到 mock。默认不要打开，避免假数据污染真实计划。

## 当前核心 Workflow

当前 LangGraph 节点顺序：

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
       replan -> plan_check_query_planner -> collect_plan_mcp_data -> validate_plan
       final_writer
```

循环规则：

- `validate_plan` 输出 high 或 critical 问题时，`repair_strategy_planner` 会选择 `replan`。
- `replan` 后重新进行 plan-check 查询、MCP 采集、校验。
- 达到 `max_iterations` 后进入 `final_writer`，但 `TripPlan.quality_gate.can_finalize` 可能为 `false`，表示只是临时输出。

## State 字段

`TripState` 当前保存：

- `raw_user_input`
- `use_llm`
- `mcp_backend`
- `mcp_errors`
- `mcp_cache`
- `mcp_cache_stats`
- `user_request`
- `city_route_plan`
- `pending_mcp_queries`
- `current_plan`
- `plan_versions`
- `mcp_results`
- `issues`
- `repair_strategy`
- `iteration`
- `max_iterations`
- `final_plan`

关键约定：

- `current_plan` 是当前结构化计划。
- `plan_versions` 保存重规划前的旧版本。
- `pending_mcp_queries` 是下一次要执行的 MCP 查询计划，执行后会清空。
- `mcp_results` 是累计合并后的外部数据。
- `mcp_cache` / `mcp_cache_stats` 用于减少重复 MCP 查询。
- `mcp_errors` 必须保留，不能静默吞掉外部服务失败。

## 哪些节点使用 LLM

当前只有这些节点会调用 LLM：

- `city_route_planner`：生成结构化 `CityRoutePlan`。
- `draft_day_schedule`：基于 city route 和 MCP 数据生成结构化 `TripPlan`。
- `replan`：根据校验问题和 MCP 数据修改结构化 `TripPlan`。

这些节点即使调用 LLM，也必须返回 Pydantic schema，不允许在内部传自然语言计划。

这些节点是 deterministic Python：

- `parse_request`
- `preplan_query_planner`
- `collect_mcp_data`
- `plan_check_query_planner`
- `validate_plan`
- `repair_strategy_planner`
- `final_writer`
- LangGraph routing

## 当前 Schema 思路

最重要的结构在 `backend/schemas/trip.py`。

输入：

- `TripRequest`：出发地、目的地、起止日期、旅客、住宿需求、预算、偏好、必去景点、避开项。
- `TravelerGroup`：成人、儿童、婴儿、儿童/婴儿是否占床、年龄列表。
- 儿童/婴儿会计入 `total_people`，但不一定计入 `bed_count`。

路线骨架：

- `CityRoutePlan`：城市/区域层面的停留与跨城段。
- `CityStayPlan`：某个城市停留几天、锚点景点、住宿锚点。
- `TripSegment`：出发、跨城、返程等路线骨架段。

最终计划：

- `TripPlan`：总计划，包含 `route_segments`、`accommodations`、`days`、费用、质量门禁。
- `PlanDay`：某一天的时间线。
- `TimelineItem`：每天的原子动作，只能是 `stay` 或 `move`。
- `StayDetail`：在某个地点停留，例如 sleep、meal、visit、rest、check-in、checkout。
- `MoveDetail`：从一个具体点移动到另一个具体点，包含交通方式、城市上下文、耗时、距离、费用。

当前设计重点：

- 旅行计划本质上是“在某点停留多久”和“从某点移动到某点耗时多久”的序列。
- 第一段必须从用户出发地开始。
- 最后一段必须回到用户出发地。
- 时间线上相邻项必须地点连续：上一项的终点应该等于下一项的起点。
- 如果地点发生变化，必须插入 `move`。
- 每个 `move` 和 `TripSegment` 都应填 `origin_city` / `destination_city`，用于高德查询时避免地理歧义。

## MCP 后端

Mock MCP：

- 使用 stdio 运行。
- 工具包括 `get_weather`、`get_route_time`、`get_attraction_detail`、`search_attractions`、`search_accommodation_areas`、`search_lodging_near_place`。
- mock 数据故意包含雨天、闭馆、长路线等冲突，用于测试校验和重规划闭环。

高德 MCP：

- official endpoint：`https://mcp.amap.com/mcp?key=...`
- bailian endpoint：`https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp`
- 当前内部映射：
  - `get_weather` -> `maps_weather`
  - `search_attractions` -> `maps_text_search`
  - `search_accommodation_areas` -> `maps_text_search`
  - `search_lodging_near_place` -> `maps_geo` + `maps_around_search` 或 `maps_text_search`
  - `get_attraction_detail` -> `maps_text_search` + `maps_search_detail`
  - `get_route_time` -> `maps_geo` + `maps_direction_*` / `maps_distance`

重要注意事项：

- 高德路线查询必须传 `origin_city` 和 `destination_city`，不能只传 `origin` / `destination`。
- `RouteResult` 和 cache key 已包含城市上下文，避免“同名地点但不同城市”的结果互相覆盖。
- 当前 `load_amap_tools()` 每次执行 MCP batch 时会构造新的 client/tools，不是长期单例；性能上还有优化空间。
- `backend/mcp/amap.py` 有模块级 `_GEOCODE_CACHE`，workflow state 里还有 query 级 `mcp_cache`。
- 默认不应该在 Amap 失败时自动 fallback mock；mock 的 8km/35min 结果会严重污染真实行程。

## 日志和调试

Streamlit 每次运行会写入：

```text
data/logs/travel_agent_YYYYMMDD_HHMMSS.jsonl
```

每个事件包含：

- `node`
- `summary`
- `node_duration_seconds`
- `elapsed_seconds`
- `iteration`
- `pending_queries`
- `mcp_cache_stats`
- `mcp_errors`
- `user_request`
- `city_route_plan`
- `pending_mcp_queries_detail`
- `current_plan`
- `plan_transfers`
- `plan_route_segments`
- `plan_accommodations`
- `mcp_results`
- `issues_detail`
- `quality_gate`
- `plan_versions_detail`
- `final_plan`

推荐调试方式：

- 先看 `mcp_errors`，确认外部服务是否失败。
- 再看 `mcp_results.routes`，确认路线数据是否可信。
- 再看 `plan_transfers`，确认计划里的移动段是否被 MCP 数据覆盖。
- 再看 `issues_detail`，确认 validator 是否识别出错误。
- 最后看 `plan_versions_detail`，确认 replan 是否真的修复了问题。

PowerShell 终端可能显示中文乱码。日志文件本身按 UTF-8 写入，优先通过 Streamlit 日志可视化或 Python `json.loads` 读取。

## 当前测试状态

最近一次已知完整测试结果：

```text
45 passed, 3 skipped
```

其中 live Amap 测试默认 skip，需要 `RUN_AMAP_MCP_LIVE_TESTS=1` 才会真实调用高德。

重要测试：

- `tests/test_schemas.py`：旅客/住宿/时间线 schema 校验。
- `tests/test_workflow.py`：workflow、缓存、拓扑校验、Amap 失败不 fallback mock 等。
- `tests/test_llm_config.py`：LLM JSON 修复，例如 `taxi + train`、`shopping` purpose、跨午夜/重叠时间。
- `tests/test_amap_route_anomalies.py`：显式城市上下文下测试太原、五台山、北京之间的异常路线。
- `tests/test_amap_parsing.py` / `tests/test_amap_mcp.py`：高德解析和工具连接相关测试。

## 基于最新日志发现的问题

最新日志文件：

```text
data/logs/travel_agent_20260619_213633.jsonl
```

这次运行的输入大致是：

- 北京 -> 山西
- 2026-06-26 到 2026-06-28
- 必去：五台山

运行结果：

- 总耗时约 `640.72s`。
- 最终进入 `final_writer`，但 `quality_gate.can_finalize=false`。
- 最终仍有 1 个 high 问题和 2 个 medium 问题。
- 最终输出是 provisional，不应视为可用旅行计划。

主要耗时：

- `city_route_planner`：约 23.93s。
- `collect_preplan_mcp_data`：约 49.23s。
- `draft_day_schedule`：约 123.15s。
- 第 1 次 `replan`：约 126.13s。
- 第 2 次 `replan`：约 137.09s。
- 第 3 次 `replan`：约 133.21s。
- `collect_plan_mcp_data` 第一次：约 41.59s。

### 1. 高德路线数据仍然异常

最新日志中仍出现明显错误路线：

- 太原 -> 五台山：`1243.29 km / 768 min`
- 五台山 -> 北京：`995.39 km / 611 min`
- 五台山景区内若干本地移动：统一变成 `8.0 km / 35 min`

这些结果不可信。项目之前的 live diagnostic 在显式传入城市上下文后，太原和五台山之间应接近 `227 km / 304 min` 级别，而不是 1243 km。

可能原因：

- 日志生成时仍有部分 query 缺失或错误使用 `origin_city` / `destination_city`。
- 五台山作为景区、县名、站点名、城市上下文混用，导致 geocode 命中错误地点。
- Amap MCP 报错后 fallback 到 mock，把 mock 的 `8 km / 35 min` 写进结果。
- `destination_city` 曾使用“五台山”这类非城市值，而不是“忻州”。

当前代码已经加强城市上下文和 cache key，并且测试要求 Amap 失败时默认不 fallback mock；需要重新生成一次最新日志验证这些改动是否已经生效。

### 2. Amap MCP 调用失败污染了计划

最新日志里有：

```text
Amap MCP failed, fell back to mock MCP: API 调用失败：OVER_DIRECTION_RANGE
```

问题在于 fallback mock 会让系统继续运行，但会把 mock 路线作为真实路线使用。对于真实旅游计划，这是危险行为。

当前代码方向应该是：

- 默认记录 `mcp_errors`。
- 跳过失败 batch 或只保留已有真实数据。
- 只有显式设置 `TRAVEL_AGENT_ALLOW_MOCK_MCP_FALLBACK=true` 时才允许 fallback。

README 中仍有“失败后 fallback mock”的旧描述，后续需要同步更新 README，避免误导。

### 3. 必去景点匹配仍然不够好

最新日志最终仍报：

```text
missing_must_visit: Must-visit place 五台山 is not present in the timeline.
```

但计划中可能安排了显通寺、菩萨顶、黛螺顶等五台山景区内景点。Validator 当前更偏向字符串匹配，无法稳定识别“父级景区已通过子景点覆盖”。

需要改进：

- 建立景区层级/别名映射，例如“五台山”包含“显通寺、菩萨顶、黛螺顶、塔院寺、殊像寺”等。
- MCP 景点详情返回后，应保留 parent scenic area 或 canonical place name。
- must_visit 校验不能只做简单包含关系。

### 4. 住宿位置仍然不可靠

最新日志中住宿候选看起来被查到了太原的一批民宿，但 anchor 是“五台山风景区入口附近”。这些候选的 `location` 为空，`distance_to_anchor_km=0.0`，导致排序和“离景点近”的判断没有意义。

问题来源：

- 高德 POI 结果可能没有 `location` 字段。
- 当前 lodging parser 没有强制二次 detail 查询或过滤无坐标结果。
- city/anchor 如果错，住宿搜索会在错误城市返回候选。

需要改进：

- 对住宿 POI 补充 `maps_search_detail` 获取坐标。
- 没有坐标的 lodging 不应被当作距离 0。
- 对住宿候选做城市、行政区、距离阈值过滤。
- 对五台山这类景区，住宿城市上下文应优先使用“忻州/五台县/台怀镇”而不是太原。

### 5. 交通信息不够具体

当前 schema 支持：

- `station_or_terminal`
- `train_or_flight_number`
- `booking_notes`

但高德 MCP 不能可靠提供火车班次、车站、余票、真实发到时间。日志里出现的“北京北站 -> 太原南站”等信息可能是 LLM 猜测，不是票务数据。

需要改进：

- 接入独立的火车/航班/票务数据源。
- 在没有真实票源时，明确标记为“待订票源确认”，不能伪装成确定班次。
- `final_writer` 应把此类信息列为用户需自行确认项。

### 6. 重规划质量仍然弱

最新日志中校验问题数量从 11 降到 5，再降到 4，最后降到 3，但仍未完全修复。

问题表现：

- replan 耗时很长，但修复幅度有限。
- LLM 有时只是局部改文案，没有真正修正路线/住宿/拓扑。
- 如果 MCP 数据本身被污染，replan 会基于错误数据越改越偏。

需要改进：

- 在 replan 前先判断 MCP 数据可信度。
- 对 critical 拓扑问题使用 deterministic repair，而不是完全交给 LLM。
- 将“计划生成”和“可行性求解”更多转为约束式程序逻辑，例如先固定跨城段，再填每日 local timeline。

### 7. 计划拓扑约束已经加入，但需要新日志验证

当前 validator 已检查：

- 第一项必须从用户出发地开始。
- 最后一项必须回到用户出发地。
- 相邻 timeline 项的地点必须连续。
- 回程后不能继续出现目的地活动。

最新日志仍然反映了一些旧问题和旧数据污染。需要用当前代码重新运行北京 -> 山西案例，确认拓扑校验和 no-fallback 行为是否真的生效。

### 8. 性能瓶颈明确

最新日志显示性能瓶颈主要有两类：

- LLM 生成/重规划：一次 120-140 秒级。
- MCP 采集：前两轮几十秒级。

已经有 query cache，但该日志中首次运行 cache 命中为 0。后续可以优化：

- 缩小传给 LLM 的 MCP payload，只传候选摘要和关键路线。
- 对 replan 使用更小的 delta prompt，而不是每次塞完整计划和完整 MCP 结果。
- 对 Amap 做更细粒度缓存，包括 geocode、route、POI detail。
- 对稳定 POI 查询持久化到本地缓存文件。
- 高德并发默认 1 是为了避开 QPS 限制，提速前必须先处理限流和重试策略。

## 后续推荐优先级

1. 重新跑一次北京 -> 山西 -> 五台山案例，确认当前代码的 no-fallback、城市上下文和 cache key 是否生效。
2. 修正 README 中关于 Amap 失败 fallback mock 的陈旧描述。
3. 做 POI canonicalization：景点名、城市、景区父子关系、别名统一。
4. 对住宿候选增加坐标补全、距离计算和无坐标过滤。
5. 把跨城路线作为硬约束先确定，再让 LLM 只填局部日程。
6. 对 critical 拓扑错误实现 deterministic repair 或直接阻断 finalize。
7. 引入真实火车/航班数据源，否则不要输出确定班次。
8. 压缩 LLM 输入和 MCP payload，减少 120 秒级规划调用。

## 后续 Agent 工作原则

- 先读 `backend/schemas/trip.py`，再改节点。
- 所有内部数据必须是 Pydantic schema，不要让自由文本在节点之间传递。
- 涉及真实路线时，必须保留 `origin_city` / `destination_city`。
- 不能把 mock 数据当作真实高德结果混用。
- 外部 API 错误必须进入 `mcp_errors` 和日志。
- 修改 workflow 后同步测试 `tests/test_workflow.py`。
- 修改 schema 后同步 Streamlit 展示、LLM prompt、mock MCP、tests。
- 计划质量优先于演示效果；如果无法保证可行，应通过 `quality_gate` 明确标记不可最终化。
