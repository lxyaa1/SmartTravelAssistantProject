from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.graph.workflow import build_workflow
from backend.schemas.trip import McpResults, TripPlan, ValidationIssue


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "name": "Shanghai family Hangzhou weekend",
        "raw_user_input": {
            "origin": "Shanghai",
            "destination": "Hangzhou",
            "start_date": "2026-07-01",
            "end_date": "2026-07-03",
            "travelers": {
                "adults": 2,
                "children": 1,
                "children_need_bed": 0,
                "children_ages": [6],
            },
            "budget_level": "medium",
            "preferences": ["culture", "food", "relaxed", "family friendly"],
            "must_visit": ["West Lake"],
            "avoid": ["late night activities", "overly packed schedule"],
        },
    },
    {
        "name": "Beijing to Shanxi Wutai Mountain",
        "raw_user_input": {
            "origin": "北京",
            "destination": "山西",
            "start_date": "2026-06-29",
            "end_date": "2026-07-01",
            "travelers": {
                "adults": 3,
                "children": 0,
                "children_need_bed": 0,
                "children_ages": [],
            },
            "budget_level": "medium",
            "preferences": ["culture", "food", "relaxed", "family friendly"],
            "must_visit": ["五台山"],
            "avoid": ["late night activities", "overly packed schedule"],
        },
    },
    {
        "name": "Shanghai to Suzhou culture day",
        "raw_user_input": {
            "origin": "Shanghai",
            "destination": "Suzhou",
            "start_date": "2026-07-04",
            "end_date": "2026-07-05",
            "travelers": {
                "adults": 2,
                "children": 0,
                "children_need_bed": 0,
                "children_ages": [],
            },
            "budget_level": "medium",
            "preferences": ["culture", "garden", "food", "walkable"],
            "must_visit": ["Humble Administrator's Garden"],
            "avoid": ["long transfers"],
        },
    },
]


EXTENDED_CASES: list[dict[str, Any]] = [
    {
        "name": "Beijing family Chengdu pandas",
        "raw_user_input": {
            "origin": "北京",
            "destination": "成都",
            "start_date": "2026-08-10",
            "end_date": "2026-08-13",
            "travelers": {
                "adults": 2,
                "children": 2,
                "children_need_bed": 1,
                "children_ages": [5, 9],
            },
            "budget_level": "medium",
            "preferences": ["family friendly", "food", "relaxed", "animals"],
            "must_visit": ["成都大熊猫繁育研究基地"],
            "avoid": ["late night activities", "overly packed schedule"],
        },
    },
    {
        "name": "Guangzhou to Guilin river weekend",
        "raw_user_input": {
            "origin": "广州",
            "destination": "桂林",
            "start_date": "2026-09-05",
            "end_date": "2026-09-07",
            "travelers": {
                "adults": 2,
                "children": 0,
                "children_need_bed": 0,
                "children_ages": [],
            },
            "budget_level": "medium",
            "preferences": ["nature", "local food", "relaxed"],
            "must_visit": ["象鼻山", "漓江景区"],
            "avoid": ["long transfers"],
        },
    },
    {
        "name": "Shanghai to Huangshan hiking",
        "raw_user_input": {
            "origin": "上海",
            "destination": "黄山",
            "start_date": "2026-10-02",
            "end_date": "2026-10-05",
            "travelers": {
                "adults": 2,
                "children": 0,
                "children_need_bed": 0,
                "children_ages": [],
            },
            "budget_level": "medium",
            "preferences": ["nature", "hiking", "scenic views"],
            "must_visit": ["黄山风景区"],
            "avoid": ["overly packed schedule", "unsafe bad-weather hikes"],
        },
    },
    {
        "name": "Beijing to Nanjing history",
        "raw_user_input": {
            "origin": "北京",
            "destination": "南京",
            "start_date": "2026-07-10",
            "end_date": "2026-07-12",
            "travelers": {
                "adults": 2,
                "children": 0,
                "children_need_bed": 0,
                "children_ages": [],
            },
            "budget_level": "medium",
            "preferences": ["history", "museum", "culture"],
            "must_visit": ["南京博物院", "中山陵"],
            "avoid": ["long transfers"],
        },
    },
    {
        "name": "Shenzhen to Xiamen island",
        "raw_user_input": {
            "origin": "深圳",
            "destination": "厦门",
            "start_date": "2026-08-21",
            "end_date": "2026-08-23",
            "travelers": {
                "adults": 3,
                "children": 0,
                "children_need_bed": 0,
                "children_ages": [],
            },
            "budget_level": "medium",
            "preferences": ["island", "food", "walkable", "culture"],
            "must_visit": ["鼓浪屿"],
            "avoid": ["overly packed schedule"],
        },
    },
]


def main() -> None:
    args = _parse_args()
    selected_cases = _select_cases(args.case, args.suite)
    report = _run_cases(
        cases=selected_cases,
        backend=args.backend,
        use_llm=args.use_llm,
        max_iterations=args.max_iterations,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html_report(report), encoding="utf-8")
    print(f"Report written to: {output_path.resolve()}")
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple travel-planning cases and render an HTML report.")
    parser.add_argument("--backend", choices=["mock", "amap"], default="mock")
    parser.add_argument("--suite", choices=["default", "extended", "all"], default="default")
    parser.add_argument("--use-llm", action="store_true", help="Call DashScope/Qwen for LLM-backed nodes.")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument(
        "--output",
        default=f"data/reports/travel_plan_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only cases whose name contains this text. Can be provided multiple times.",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated HTML report in the default browser.")
    return parser.parse_args()


def _select_cases(filters: list[str] | None, suite: str) -> list[dict[str, Any]]:
    cases = {
        "default": DEFAULT_CASES,
        "extended": EXTENDED_CASES,
        "all": [*DEFAULT_CASES, *EXTENDED_CASES],
    }[suite]
    if not filters:
        return cases
    lowered = [item.lower() for item in filters]
    selected = [
        case
        for case in cases
        if any(token in case["name"].lower() for token in lowered)
    ]
    if not selected:
        raise SystemExit(f"No cases matched: {', '.join(filters)}")
    return selected


def _run_cases(
    cases: list[dict[str, Any]],
    backend: str,
    use_llm: bool,
    max_iterations: int,
) -> dict[str, Any]:
    workflow = build_workflow()
    results = []
    started_at = datetime.now()
    for case in cases:
        start = perf_counter()
        initial_state = {
            "raw_user_input": case["raw_user_input"],
            "mcp_backend": backend,
            "use_llm": use_llm,
            "max_iterations": max_iterations,
        }
        try:
            state = workflow.invoke(initial_state)
            elapsed = perf_counter() - start
            results.append(_case_success(case=case, state=state, elapsed_seconds=elapsed))
        except Exception as exc:
            elapsed = perf_counter() - start
            results.append(
                {
                    "name": case["name"],
                    "raw_user_input": case["raw_user_input"],
                    "elapsed_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "generated_at": started_at.isoformat(timespec="seconds"),
        "backend": backend,
        "use_llm": use_llm,
        "max_iterations": max_iterations,
        "cases": results,
    }


def _case_success(case: dict[str, Any], state: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    plan: TripPlan = state["current_plan"]
    issues: list[ValidationIssue] = state.get("issues", [])
    mcp_results: McpResults = state.get("mcp_results", McpResults())
    return {
        "name": case["name"],
        "raw_user_input": case["raw_user_input"],
        "elapsed_seconds": elapsed_seconds,
        "iteration": state.get("iteration", 0),
        "can_finalize": plan.quality_gate.can_finalize,
        "quality_gate": _jsonable(plan.quality_gate),
        "issue_count": len(issues),
        "issues": [_jsonable(issue) for issue in issues],
        "mcp_error_count": len(state.get("mcp_errors", [])),
        "mcp_errors": list(state.get("mcp_errors", [])),
        "mcp_cache_stats": dict(state.get("mcp_cache_stats", {})),
        "route_count": len(mcp_results.routes),
        "lodging_count": len(mcp_results.lodging),
        "weather_count": len(mcp_results.weather),
        "attraction_count": len(mcp_results.attractions),
        "plan": _jsonable(plan),
        "final_plan": _jsonable(state.get("final_plan")),
    }


def _render_html_report(report: dict[str, Any]) -> str:
    body = [
        "<!doctype html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>TravelAgent Batch Report</title>",
        _style(),
        "</head>",
        "<body>",
        "<main>",
        "<header>",
        "<h1>TravelAgent 批量规划测试报告</h1>",
        f"<p>生成时间：{_esc(report['generated_at'])} | MCP：{_esc(report['backend'])} | "
        f"LLM：{_esc(str(report['use_llm']))} | 最大重规划：{_esc(str(report['max_iterations']))}</p>",
        "</header>",
        _render_case_index(report["cases"]),
    ]
    for index, case in enumerate(report["cases"], start=1):
        body.append(_render_case(case, index))
    body.extend(["</main>", "</body>", "</html>"])
    return "\n".join(body)


def _render_case_index(cases: list[dict[str, Any]]) -> str:
    rows = []
    for index, case in enumerate(cases, start=1):
        status = "ERROR" if case.get("error") else ("OK" if case.get("can_finalize") else "BLOCKED")
        rows.append(
            "<tr>"
            f"<td><a href=\"#case-{index}\">{_esc(case['name'])}</a></td>"
            f"<td><span class=\"badge {status.lower()}\">{status}</span></td>"
            f"<td>{case.get('issue_count', '-')}</td>"
            f"<td>{case.get('mcp_error_count', '-')}</td>"
            f"<td>{case.get('route_count', '-')}</td>"
            f"<td>{case.get('elapsed_seconds', 0):.1f}s</td>"
            "</tr>"
        )
    return (
        "<section class=\"panel\">"
        "<h2>总览</h2>"
        "<table><thead><tr><th>案例</th><th>状态</th><th>问题</th><th>MCP 错误</th><th>路线</th><th>耗时</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def _render_case(case: dict[str, Any], case_number: int) -> str:
    case_id = ""
    if "error" in case:
        return (
            "<section class=\"panel error-panel\">"
            f"<h2>{_esc(case['name'])}</h2>"
            "<h3>运行错误</h3>"
            f"<pre>{_esc(case['error'])}</pre>"
            "<h3>输入</h3>"
            f"<pre>{_esc(json.dumps(case['raw_user_input'], ensure_ascii=False, indent=2))}</pre>"
            "</section>"
        )

    index_block = (
        f"<div class=\"metrics\">"
        f"<div><b>{'可最终化' if case['can_finalize'] else '不可最终化'}</b><span>质量门禁</span></div>"
        f"<div><b>{case['issue_count']}</b><span>问题</span></div>"
        f"<div><b>{case['mcp_error_count']}</b><span>MCP 错误</span></div>"
        f"<div><b>{case['route_count']}</b><span>路线结果</span></div>"
        f"<div><b>{case['elapsed_seconds']:.1f}s</b><span>耗时</span></div>"
        "</div>"
    )
    plan = case["plan"]
    case_id = f"case-{case_number}"
    return (
        f"<section id=\"{case_id}\" class=\"panel\">"
        f"<h2>{_esc(case['name'])}</h2>"
        f"{index_block}"
        f"<p class=\"quality\">{_esc(case['quality_gate'].get('reason', ''))}</p>"
        "<details open><summary>问题列表</summary>"
        f"{_render_issues(case['issues'])}</details>"
        "<details><summary>MCP 错误</summary>"
        f"{_render_list(case['mcp_errors'])}</details>"
        "<details><summary>Cache 统计</summary>"
        f"<pre>{_esc(json.dumps(case['mcp_cache_stats'], ensure_ascii=False, indent=2))}</pre></details>"
        "<details><summary>路线骨架</summary>"
        f"{_render_route_segments(plan)}</details>"
        "<details open><summary>每日时间线</summary>"
        f"{_render_days(plan)}</details>"
        "<details><summary>最终文本</summary>"
        f"<pre>{_esc((case.get('final_plan') or {}).get('content', ''))}</pre></details>"
        "<details><summary>原始输入</summary>"
        f"<pre>{_esc(json.dumps(case['raw_user_input'], ensure_ascii=False, indent=2))}</pre></details>"
        "</section>"
    )


def _render_issues(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return "<p class=\"muted\">无未解决问题。</p>"
    rows = []
    for issue in issues:
        rows.append(
            "<tr>"
            f"<td>{_esc(issue.get('severity', ''))}</td>"
            f"<td>{_esc(issue.get('issue_type', ''))}</td>"
            f"<td>{_esc(str(issue.get('day') or ''))}</td>"
            f"<td>{_esc(', '.join(issue.get('locations', [])))}</td>"
            f"<td>{_esc(issue.get('reason', ''))}</td>"
            f"<td>{_esc(issue.get('suggested_action', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>严重度</th><th>类型</th><th>Day</th><th>地点</th><th>原因</th><th>建议</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_route_segments(plan: dict[str, Any]) -> str:
    segments = plan.get("route_segments", [])
    if not segments:
        return "<p class=\"muted\">无路线骨架。</p>"
    rows = []
    for segment in segments:
        rows.append(
            "<tr>"
            f"<td>{_esc(str(segment.get('sequence', '')))}</td>"
            f"<td>{_esc(segment.get('segment_type', ''))}</td>"
            f"<td>{_esc(segment.get('origin', ''))} -> {_esc(segment.get('destination', ''))}</td>"
            f"<td>{_esc(segment.get('mode', ''))}</td>"
            f"<td>{_esc(str(segment.get('estimated_duration_minutes', '')))}</td>"
            f"<td>{_esc(str(segment.get('estimated_distance_km', '')))}</td>"
            f"<td>{_esc(segment.get('notes', '') or segment.get('booking_notes', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>类型</th><th>路线</th><th>交通</th><th>分钟</th><th>km</th><th>备注</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_days(plan: dict[str, Any]) -> str:
    blocks = []
    for day in plan.get("days", []):
        rows = []
        for item in day.get("timeline", []):
            stay = item.get("stay") or {}
            move = item.get("move") or {}
            place_or_route = stay.get("place_name") or f"{move.get('origin', '')} -> {move.get('destination', '')}"
            rows.append(
                "<tr>"
                f"<td>{_esc(str(item.get('sequence', '')))}</td>"
                f"<td>{_esc(item.get('start_time', ''))} - {_esc(item.get('end_time', ''))}</td>"
                f"<td>{_esc(item.get('item_type', ''))}</td>"
                f"<td>{_esc(stay.get('purpose') or move.get('purpose') or '')}</td>"
                f"<td>{_esc(place_or_route)}</td>"
                f"<td>{_esc(move.get('mode', ''))}</td>"
                f"<td>{_esc(str(stay.get('duration_minutes') or move.get('duration_minutes') or item.get('duration_minutes', '')))}</td>"
                f"<td>{_esc(str(move.get('distance_km', '')))}</td>"
                "</tr>"
            )
        blocks.append(
            f"<h3>Day {day.get('day')} | {_esc(day.get('date', ''))} | {_esc(day.get('city', ''))}</h3>"
            f"<p class=\"muted\">住宿：{_esc(day.get('accommodation_area') or '')}</p>"
            "<table><thead><tr><th>#</th><th>时间</th><th>类型</th><th>目的</th><th>地点/路线</th><th>交通</th><th>分钟</th><th>km</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return "\n".join(blocks)


def _render_list(items: list[str]) -> str:
    if not items:
        return "<p class=\"muted\">无。</p>"
    return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _style() -> str:
    return """
<style>
:root { color-scheme: light; font-family: Arial, "Microsoft YaHei", sans-serif; }
body { margin: 0; background: #f5f6f8; color: #20242a; }
main { max-width: 1280px; margin: 0 auto; padding: 24px; }
header { margin-bottom: 18px; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin: 0 0 14px; font-size: 21px; }
h3 { margin: 18px 0 8px; font-size: 16px; }
.panel { background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 18px; margin: 16px 0; }
.error-panel { border-color: #d64545; }
.metrics { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin: 12px 0; }
.metrics div { border: 1px solid #dfe3e8; border-radius: 6px; padding: 10px; background: #fafbfc; }
.metrics b { display: block; font-size: 18px; margin-bottom: 4px; }
.metrics span, .muted { color: #69707a; }
.quality { border-left: 4px solid #7c8798; padding-left: 10px; color: #414852; }
.badge { display: inline-block; min-width: 72px; text-align: center; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge.ok { background: #dcfce7; color: #166534; }
.badge.blocked { background: #fef3c7; color: #92400e; }
.badge.error { background: #fee2e2; color: #991b1b; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 13px; }
th, td { border: 1px solid #dfe3e8; padding: 7px 8px; vertical-align: top; }
th { background: #f0f2f5; text-align: left; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #111827; color: #f9fafb; border-radius: 6px; padding: 12px; font-size: 12px; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: 700; margin-bottom: 8px; }
a { color: #195bd8; }
@media (max-width: 900px) { .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); } main { padding: 14px; } }
</style>
"""


if __name__ == "__main__":
    main()
