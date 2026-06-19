from __future__ import annotations

import os
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st
from pydantic import BaseModel

from backend.graph.workflow import build_workflow
from backend.schemas.trip import McpResults, TripPlan, ValidationIssue


st.set_page_config(page_title="TravelAgent", page_icon="T", layout="wide")
LOG_DIR = Path("data/logs")
UI_STATE_VERSION = 6


def main() -> None:
    _clear_stale_session_state()
    _apply_style()

    st.title("\u672c\u5730\u65c5\u6e38\u52a9\u624b")

    col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1, 1])
    with col_a:
        origin = st.text_input("\u51fa\u53d1\u5730", value="\u4e0a\u6d77")
        adults = st.number_input("\u6210\u4eba", min_value=1, max_value=20, value=2, step=1)
    with col_b:
        destination = st.text_input("\u76ee\u7684\u5730", value="\u676d\u5dde")
        children = st.number_input("\u513f\u7ae5", min_value=0, max_value=20, value=1, step=1)
    with col_c:
        start_date = st.date_input("\u5f00\u59cb\u65e5\u671f", value=date.today() + timedelta(days=7))
        children_need_bed = st.number_input(
            "\u513f\u7ae5\u5360\u5e8a",
            min_value=0,
            max_value=int(children),
            value=0,
            step=1,
            key=f"children_need_bed_{int(children)}",
        )
    with col_d:
        end_date = st.date_input("\u7ed3\u675f\u65e5\u671f", value=date.today() + timedelta(days=9))
        budget_level = st.selectbox("\u9884\u7b97", options=["low", "medium", "high"], index=1)

    col_e, col_f, col_g = st.columns([1.2, 1.2, 1])
    with col_e:
        preferences_text = st.text_area("\u504f\u597d", value="culture, food, relaxed, family friendly", height=90)
        must_visit_text = st.text_area("\u5fc5\u53bb\u666f\u70b9", value="\u897f\u6e56", height=80)
    with col_f:
        avoid_text = st.text_area("\u907f\u5f00\u9879", value="late night activities, overly packed schedule", height=90)
        children_ages = _render_child_age_inputs(int(children))
    with col_g:
        backend = st.radio("MCP", options=["mock", "amap"], horizontal=True)
        amap_provider = st.selectbox("\u9ad8\u5fb7\u6765\u6e90", options=["official", "auto", "bailian"], index=0)
        use_llm = st.checkbox("LLM", value=False)
        max_iterations = st.number_input("\u6700\u5927\u91cd\u89c4\u5212", min_value=0, max_value=5, value=3, step=1)

    submitted = st.button("\u751f\u6210\u8ba1\u5212", type="primary", use_container_width=True)

    if submitted:
        raw_user_input = {
            "origin": origin,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "travelers": {
                "adults": int(adults),
                "children": int(children),
                "children_need_bed": int(children_need_bed),
                "children_ages": children_ages,
            },
            "budget_level": budget_level,
            "preferences": _parse_text_list(preferences_text),
            "must_visit": _parse_text_list(must_visit_text),
            "avoid": _parse_text_list(avoid_text),
        }

        progress_container = st.container()
        with st.spinner("\u8fd0\u884c\u4e2d"):
            try:
                with _temporary_env("TRAVEL_AGENT_AMAP_PROVIDER", amap_provider):
                    result, events, log_path = _run_workflow_streaming(
                        raw_user_input=raw_user_input,
                        backend=backend,
                        use_llm=use_llm,
                        max_iterations=int(max_iterations),
                        progress_container=progress_container,
                    )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                return

        st.session_state["last_result"] = result
        st.session_state["last_events"] = events
        st.session_state["last_log_path"] = str(log_path)

    result = st.session_state.get("last_result")
    if result:
        _render_result(result)

    st.divider()
    _render_log_file_viewer()


def _render_child_age_inputs(children: int) -> list[int]:
    if children <= 0:
        st.caption("\u513f\u7ae5\u6570\u91cf\u4e3a 0\uff0c\u4e0d\u9700\u586b\u5199\u5e74\u9f84\u3002")
        return []

    st.caption("\u513f\u7ae5\u5e74\u9f84")
    ages: list[int] = []
    columns = st.columns(min(children, 4))
    for index in range(children):
        with columns[index % len(columns)]:
            age = st.number_input(
                f"\u513f\u7ae5 {index + 1}",
                min_value=2,
                max_value=17,
                value=6,
                step=1,
                key=f"child_age_{index}",
            )
        ages.append(int(age))
    return ages


def _run_workflow_streaming(
    raw_user_input: dict[str, Any],
    backend: str,
    use_llm: bool,
    max_iterations: int,
    progress_container,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    workflow = build_workflow()
    initial_state = {
        "raw_user_input": raw_user_input,
        "mcp_backend": backend,
        "use_llm": use_llm,
        "max_iterations": max_iterations,
    }
    events: list[dict[str, Any]] = []
    log_path = _new_log_path()
    latest_state: dict[str, Any] = {}

    status_box = progress_container.empty()
    plan_box = progress_container.empty()
    issue_box = progress_container.empty()
    started_at = perf_counter()
    last_event_at = started_at

    for index, update in enumerate(workflow.stream(initial_state, stream_mode="updates"), start=1):
        for node_name, node_state in update.items():
            if not isinstance(node_state, dict):
                continue
            now = perf_counter()
            node_duration_seconds = now - last_event_at
            elapsed_seconds = now - started_at
            last_event_at = now
            latest_state = node_state
            event = _workflow_event(
                index=index,
                node_name=node_name,
                state=node_state,
                node_duration_seconds=node_duration_seconds,
                elapsed_seconds=elapsed_seconds,
            )
            events.append(event)
            _append_log(log_path, event)
            status_box.info(
                f"{event['step']}. {event['node']}: {event['summary']} "
                f"({event['node_duration_seconds']:.2f}s)"
            )
            if node_state.get("current_plan"):
                with plan_box.container():
                    st.caption("\u5f53\u524d\u8ba1\u5212")
                    _render_plan_snapshot(node_state["current_plan"])
            if node_state.get("issues"):
                with issue_box.container():
                    st.caption("\u5f53\u524d\u95ee\u9898")
                    st.dataframe(
                        [issue.model_dump(mode="json") for issue in node_state["issues"]],
                        hide_index=True,
                        use_container_width=True,
                    )

    if not latest_state.get("final_plan"):
        latest_state = workflow.invoke(initial_state)
    return latest_state, events, log_path


def _render_result(result: dict[str, Any]) -> None:
    plan: TripPlan = result["current_plan"]
    issues: list[ValidationIssue] = result.get("issues", [])
    mcp_results: McpResults = result.get("mcp_results", McpResults())
    mcp_errors = result.get("mcp_errors", [])

    metric_cols = st.columns(5)
    metric_cols[0].metric("\u5929\u6570", len(plan.days))
    metric_cols[1].metric("\u91cd\u89c4\u5212", result.get("iteration", 0))
    metric_cols[2].metric("\u9884\u7b97", f"{plan.total_estimated_cost:.0f} {plan.currency}")
    metric_cols[3].metric("\u95ee\u9898", len(issues))
    metric_cols[4].metric("MCP \u8b66\u544a", len(mcp_errors))

    if mcp_errors:
        with st.expander("MCP \u8b66\u544a", expanded=True):
            for error in mcp_errors:
                st.warning(error)
    if not plan.quality_gate.can_finalize:
        st.warning(plan.quality_gate.reason)

    itinerary_tab, issues_tab, mcp_tab, log_tab, final_tab = st.tabs(
        ["\u884c\u7a0b", "\u95ee\u9898", "MCP \u6570\u636e", "\u884c\u52a8\u65e5\u5fd7", "\u6700\u7ec8\u6587\u672c"]
    )
    with itinerary_tab:
        _render_itinerary(plan)
    with issues_tab:
        _render_issues(issues)
    with mcp_tab:
        _render_mcp_results(mcp_results)
    with log_tab:
        _render_event_log()
    with final_tab:
        st.markdown(result["final_plan"].content)


def _render_itinerary(plan: TripPlan) -> None:
    st.subheader(plan.title)
    if plan.route_segments:
        st.markdown("### \u51fa\u884c\u8def\u7ebf")
        st.dataframe(
            [
                {
                    "\u987a\u5e8f": segment.sequence,
                    "\u7c7b\u578b": segment.segment_type.value,
                    "\u51fa\u53d1": segment.origin,
                    "\u5230\u8fbe": segment.destination,
                    "\u4ea4\u901a": segment.mode.value,
                    "\u51fa\u53d1\u65f6\u95f4": _format_date_time(segment.departure_date, segment.departure_time),
                    "\u5230\u8fbe\u65f6\u95f4": _format_date_time(segment.arrival_date, segment.arrival_time),
                    "\u8017\u65f6\u5206\u949f": segment.estimated_duration_minutes,
                    "\u8ddd\u79bbkm": segment.estimated_distance_km,
                    "\u73ed\u6b21": segment.train_or_flight_number or "\u5f85\u8ba2\u7968\u6e90\u786e\u8ba4",
                    "\u5907\u6ce8": segment.booking_notes or segment.notes,
                }
                for segment in plan.route_segments
            ],
            hide_index=True,
            use_container_width=True,
        )
    if plan.accommodations:
        st.markdown("### \u4f4f\u5bbf")
        st.dataframe(
            [
                {
                    "\u9152\u5e97/\u6c11\u5bbf": stay.hotel_name,
                    "\u57ce\u5e02": stay.city,
                    "\u533a\u57df": stay.area,
                    "\u5730\u5740": stay.address,
                    "\u5165\u4f4f": stay.check_in_date.isoformat(),
                    "\u79bb\u5e97": stay.check_out_date.isoformat(),
                    "\u5e8a\u4f4d": stay.bed_count,
                    "\u9760\u8fd1": ", ".join(stay.nearby_anchor_places),
                    "\u539f\u56e0": stay.reason,
                }
                for stay in plan.accommodations
            ],
            hide_index=True,
            use_container_width=True,
        )
    for day in plan.days:
        st.markdown(f"### Day {day.day} - {day.date} - {day.city}")
        st.dataframe(
            [_timeline_item_row(day, item) for item in day.timeline],
            hide_index=True,
            use_container_width=True,
        )
        if day.accommodation_area:
            st.caption(f"\u4f4f\u5bbf\u533a\u57df\uff1a{day.accommodation_area}")
        if day.daily_notes:
            st.info(day.daily_notes)


def _timeline_item_row(day, item) -> dict[str, Any]:
    if item.move:
        return {
            "\u987a\u5e8f": item.sequence,
            "\u65f6\u95f4": f"{item.start_time.strftime('%H:%M')}-{item.end_time.strftime('%H:%M')}",
            "\u7c7b\u578b": "move",
            "\u76ee\u7684": item.move.purpose.value,
            "\u5730\u70b9/\u8def\u7ebf": f"{item.move.origin} -> {item.move.destination}",
            "\u4ea4\u901a": item.move.mode.value,
            "\u8017\u65f6\u5206\u949f": item.move.duration_minutes or item.duration_minutes,
            "\u8ddd\u79bbkm": item.move.distance_km,
            "\u8d39\u7528": item.move.estimated_cost,
            "\u5907\u6ce8": item.move.notes or item.notes,
        }
    if item.stay:
        return {
            "\u987a\u5e8f": item.sequence,
            "\u65f6\u95f4": f"{item.start_time.strftime('%H:%M')}-{item.end_time.strftime('%H:%M')}",
            "\u7c7b\u578b": "stay",
            "\u76ee\u7684": item.stay.purpose.value,
            "\u5730\u70b9/\u8def\u7ebf": item.stay.place_name,
            "\u4ea4\u901a": "",
            "\u8017\u65f6\u5206\u949f": item.stay.duration_minutes or item.duration_minutes,
            "\u8ddd\u79bbkm": "",
            "\u8d39\u7528": item.stay.estimated_cost,
            "\u5907\u6ce8": item.stay.notes or item.notes,
        }
    return {
        "\u987a\u5e8f": item.sequence,
        "\u65f6\u95f4": f"{item.start_time.strftime('%H:%M')}-{item.end_time.strftime('%H:%M')}",
        "\u7c7b\u578b": item.item_type.value,
        "\u76ee\u7684": "",
        "\u5730\u70b9/\u8def\u7ebf": day.city,
        "\u4ea4\u901a": "",
        "\u8017\u65f6\u5206\u949f": item.duration_minutes,
        "\u8ddd\u79bbkm": "",
        "\u8d39\u7528": "",
        "\u5907\u6ce8": item.notes,
    }


def _render_issues(issues: list[ValidationIssue]) -> None:
    if not issues:
        st.success("\u65e0 high / critical \u672a\u89e3\u51b3\u95ee\u9898")
        return
    rows = [issue.model_dump(mode="json") for issue in issues]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_mcp_results(mcp_results: McpResults) -> None:
    weather_tab, attraction_tab, route_tab, area_tab, lodging_tab = st.tabs(
        ["\u5929\u6c14", "\u666f\u70b9", "\u8def\u7ebf", "\u4f4f\u5bbf\u533a\u57df", "\u4f4f\u5bbf\u5019\u9009"]
    )
    with weather_tab:
        st.dataframe([item.model_dump(mode="json") for item in mcp_results.weather], hide_index=True, use_container_width=True)
    with attraction_tab:
        st.dataframe([item.model_dump(mode="json") for item in mcp_results.attractions], hide_index=True, use_container_width=True)
    with route_tab:
        st.dataframe([item.model_dump(mode="json") for item in mcp_results.routes], hide_index=True, use_container_width=True)
    with area_tab:
        st.dataframe(
            [item.model_dump(mode="json") for item in mcp_results.accommodation_areas],
            hide_index=True,
            use_container_width=True,
        )
    with lodging_tab:
        st.dataframe(
            [item.model_dump(mode="json") for item in _mcp_lodging(mcp_results)],
            hide_index=True,
            use_container_width=True,
        )


def _render_plan_snapshot(plan: TripPlan) -> None:
    rows = []
    for day in plan.days:
        visit_names = [
            item.stay.place_name
            for item in day.timeline
            if item.stay and item.stay.purpose.value == "visit"
        ]
        rows.append(
            {
                "\u5929": day.day,
                "\u65e5\u671f": day.date.isoformat(),
                "\u57ce\u5e02": day.city,
                "\u4f4f\u5bbf": day.accommodation_area or "",
                "\u666f\u70b9": " -> ".join(visit_names),
                "\u4ea4\u901a\u5206\u949f": day.total_move_minutes,
                "\u7761\u7720\u5206\u949f": day.total_sleep_minutes,
                "\u8d39\u7528": day.estimated_cost,
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_event_log() -> None:
    events = st.session_state.get("last_events", [])
    log_path = st.session_state.get("last_log_path", "")
    if log_path:
        st.caption(f"\u65e5\u5fd7\u6587\u4ef6\uff1a{log_path}")
    if not events:
        st.info("\u8fd8\u6ca1\u6709\u884c\u52a8\u65e5\u5fd7")
        return
    summary_rows = [
        {
            "time": event.get("time"),
            "step": event.get("step"),
            "node": event.get("node"),
            "summary": event.get("summary"),
            "iteration": event.get("iteration"),
            "pending_queries": event.get("pending_queries"),
            "issues": event.get("issues"),
            "node_seconds": event.get("node_duration_seconds"),
            "elapsed_seconds": event.get("elapsed_seconds"),
            "last_cache_hits": event.get("mcp_cache_stats", {}).get("last_hits", 0),
            "last_cache_misses": event.get("mcp_cache_stats", {}).get("last_misses", 0),
            "cache_hits": event.get("mcp_cache_stats", {}).get("hits", 0),
            "cache_misses": event.get("mcp_cache_stats", {}).get("misses", 0),
            "cache_entries": event.get("mcp_cache_stats", {}).get("entries", 0),
            "routes": len(event.get("mcp_results", {}).get("routes", [])),
            "lodging": len(event.get("mcp_results", {}).get("lodging", [])),
        }
        for event in events
    ]
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)
    with st.expander("\u6700\u540e\u4e00\u4e2a\u8282\u70b9\u7684\u5b8c\u6574 JSON", expanded=False):
        st.json(events[-1])


def _render_log_file_viewer() -> None:
    st.subheader("\u65e5\u5fd7\u6587\u4ef6\u53ef\u89c6\u5316")
    log_files = sorted(LOG_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not log_files:
        st.caption("\u6682\u65e0\u65e5\u5fd7\u6587\u4ef6")
        return

    selected = st.selectbox(
        "\u9009\u62e9\u65e5\u5fd7",
        options=log_files,
        format_func=lambda path: f"{path.name} ({path.stat().st_size} bytes)",
        key="selected_log_file",
    )
    events = _load_log_events(selected)
    if not events:
        st.warning("\u65e5\u5fd7\u6587\u4ef6\u4e3a\u7a7a\u6216\u683c\u5f0f\u4e0d\u6b63\u786e")
        return

    timeline_rows = [
        {
            "step": event.get("step"),
            "time": event.get("time"),
            "node": event.get("node"),
            "summary": event.get("summary"),
            "issues": event.get("issues"),
            "node_seconds": event.get("node_duration_seconds"),
            "elapsed_seconds": event.get("elapsed_seconds"),
            "last_cache_hits": event.get("mcp_cache_stats", {}).get("last_hits", 0),
            "last_cache_misses": event.get("mcp_cache_stats", {}).get("last_misses", 0),
            "cache_hits": event.get("mcp_cache_stats", {}).get("hits", 0),
            "cache_misses": event.get("mcp_cache_stats", {}).get("misses", 0),
            "cache_entries": event.get("mcp_cache_stats", {}).get("entries", 0),
            "routes": len(event.get("mcp_results", {}).get("routes", [])),
            "lodging": len(event.get("mcp_results", {}).get("lodging", [])),
            "transfers": len(event.get("plan_transfers", [])),
        }
        for event in events
    ]
    st.dataframe(timeline_rows, hide_index=True, use_container_width=True)

    final_event = _latest_event_with(events, "current_plan") or events[-1]
    if "current_plan" not in final_event:
        st.info("\u8fd9\u4e2a\u65e5\u5fd7\u53ea\u6709\u6458\u8981\uff0c\u6ca1\u6709 current_plan \u8be6\u60c5\u3002\u8bf7\u91cd\u65b0\u751f\u6210\u4e00\u6b21\u8ba1\u5212\u3002")
        return

    plan = final_event.get("current_plan", {})
    issues = final_event.get("issues_detail", [])
    routes = final_event.get("mcp_results", {}).get("routes", [])
    lodging = final_event.get("mcp_results", {}).get("lodging", [])
    transfers = final_event.get("plan_transfers", [])
    route_segments = final_event.get("plan_route_segments", [])
    accommodations = final_event.get("plan_accommodations", [])

    plan_tab, route_segment_tab, accommodation_tab, transfer_tab, route_tab, lodging_tab, issue_tab, raw_tab = st.tabs(
        ["\u8ba1\u5212", "\u8def\u7ebf\u9aa8\u67b6", "\u4f4f\u5bbf", "\u4ea4\u901a\u6bb5", "MCP \u8def\u7ebf", "MCP \u4f4f\u5bbf", "\u95ee\u9898", "\u539f\u59cb JSON"]
    )
    with plan_tab:
        _render_log_plan(plan)
    with route_segment_tab:
        _render_json_rows(route_segments, "\u65e0\u8def\u7ebf\u9aa8\u67b6")
    with accommodation_tab:
        _render_json_rows(accommodations, "\u65e0\u4f4f\u5bbf\u8bb0\u5f55")
    with transfer_tab:
        _render_json_rows(transfers, "\u65e0\u4ea4\u901a\u6bb5\u8bb0\u5f55")
    with route_tab:
        _render_json_rows(routes, "\u65e0 MCP \u8def\u7ebf\u7ed3\u679c")
    with lodging_tab:
        _render_json_rows(lodging, "\u65e0 MCP \u4f4f\u5bbf\u5019\u9009")
    with issue_tab:
        _render_json_rows(issues, "\u65e0\u95ee\u9898")
    with raw_tab:
        st.json(final_event)


def _render_log_plan(plan: dict[str, Any]) -> None:
    st.caption(f"{plan.get('title', '')} | {plan.get('origin', '')} -> {plan.get('destination', '')}")
    day_rows = []
    timeline_rows = []
    for day in plan.get("days", []):
        visits = [
            (item.get("stay") or {}).get("place_name")
            for item in day.get("timeline", [])
            if (item.get("stay") or {}).get("purpose") == "visit"
        ]
        day_rows.append(
            {
                "day": day.get("day"),
                "date": day.get("date"),
                "city": day.get("city"),
                "hotel": day.get("accommodation_area"),
                "visits": " -> ".join(item for item in visits if item),
                "move_minutes": day.get("total_move_minutes"),
                "stay_minutes": day.get("total_stay_minutes"),
                "sleep_minutes": day.get("total_sleep_minutes"),
                "cost": day.get("estimated_cost"),
            }
        )
        for item in day.get("timeline", []):
            stay = item.get("stay") or {}
            move = item.get("move") or {}
            timeline_rows.append(
                {
                    "day": day.get("day"),
                    "date": day.get("date"),
                    "time": f"{item.get('start_time')} - {item.get('end_time')}",
                    "type": item.get("item_type"),
                    "purpose": stay.get("purpose") or move.get("purpose"),
                    "place_or_route": stay.get("place_name") or f"{move.get('origin', '')} -> {move.get('destination', '')}",
                    "mode": move.get("mode", ""),
                    "minutes": stay.get("duration_minutes") or move.get("duration_minutes") or "",
                    "cost": stay.get("estimated_cost") or move.get("estimated_cost") or "",
                    "notes": stay.get("notes") or move.get("notes") or item.get("notes", ""),
                }
            )
    st.dataframe(day_rows, hide_index=True, use_container_width=True)
    if timeline_rows:
        st.caption("Timeline")
        st.dataframe(timeline_rows, hide_index=True, use_container_width=True)


def _render_json_rows(rows: list[dict[str, Any]], empty_message: str) -> None:
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _load_log_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _latest_event_with(events: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if key in event:
            return event
    return None


def _format_date_time(day_value, time_value) -> str:
    if day_value and time_value:
        return f"{day_value} {time_value.strftime('%H:%M') if hasattr(time_value, 'strftime') else time_value}"
    if day_value:
        return str(day_value)
    if time_value:
        return time_value.strftime("%H:%M") if hasattr(time_value, "strftime") else str(time_value)
    return ""


def _new_log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"travel_agent_{timestamp}.jsonl"


def _append_log(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _workflow_event(
    index: int,
    node_name: str,
    state: dict[str, Any],
    node_duration_seconds: float = 0.0,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    query_plan = state.get("pending_mcp_queries")
    issues = state.get("issues", [])
    plan = state.get("current_plan")
    mcp_results = state.get("mcp_results")
    final_plan = state.get("final_plan")

    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "step": index,
        "node": node_name,
        "summary": _summarize_state(node_name, state),
        "node_duration_seconds": round(node_duration_seconds, 4),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "iteration": state.get("iteration", 0),
        "pending_queries": len(query_plan.queries) if query_plan else 0,
        "issues": len(issues),
        "plan_versions": len(state.get("plan_versions", [])),
        "state_keys": sorted(state.keys()),
        "mcp_errors": list(state.get("mcp_errors", [])),
        "mcp_cache_stats": dict(state.get("mcp_cache_stats", {})),
    }
    if state.get("raw_user_input"):
        event["raw_user_input"] = _to_jsonable(state["raw_user_input"])
    if state.get("user_request"):
        event["user_request"] = _to_jsonable(state["user_request"])
    if state.get("city_route_plan"):
        event["city_route_plan"] = _to_jsonable(state["city_route_plan"])
    if state.get("repair_strategy"):
        event["repair_strategy"] = _to_jsonable(state["repair_strategy"])
    if query_plan:
        event["pending_mcp_queries_detail"] = _to_jsonable(query_plan)
    if plan:
        event["current_plan"] = _to_jsonable(plan)
        event["plan_transfers"] = _extract_plan_transfers(plan)
        event["plan_route_segments"] = _to_jsonable(plan.route_segments)
        event["plan_accommodations"] = _to_jsonable(plan.accommodations)
        event["quality_gate"] = _to_jsonable(plan.quality_gate)
    if mcp_results:
        event["mcp_results"] = _to_jsonable(mcp_results)
    if issues:
        event["issues_detail"] = _to_jsonable(issues)
    if state.get("plan_versions"):
        event["plan_versions_detail"] = [_to_jsonable(plan_version) for plan_version in state["plan_versions"]]
    if final_plan:
        event["final_plan"] = _to_jsonable(final_plan)
    return event


def _summarize_state(node_name: str, state: dict[str, Any]) -> str:
    if node_name == "parse_request" and state.get("user_request"):
        request = state["user_request"]
        return f"{request.origin} -> {request.destination}, {request.start_date} to {request.end_date}"
    if "query_planner" in node_name:
        query_plan = state.get("pending_mcp_queries")
        count = len(query_plan.queries) if query_plan else 0
        return f"planned {count} MCP queries"
    if "collect" in node_name:
        results = state.get("mcp_results", McpResults())
        cache_stats = state.get("mcp_cache_stats", {})
        return (
            f"weather={len(results.weather)}, attractions={len(results.attractions)}, "
            f"routes={len(results.routes)}, areas={len(results.accommodation_areas)}, lodging={len(_mcp_lodging(results))}, "
            f"cache={cache_stats.get('last_hits', 0)}/{cache_stats.get('last_misses', 0)}"
        )
    if node_name in {"draft_day_schedule", "initial_plan", "replan"} and state.get("current_plan"):
        plan = state["current_plan"]
        return f"{len(plan.days)} days, cost {plan.total_estimated_cost:.0f} {plan.currency}"
    if node_name == "validate_plan":
        return f"found {len(state.get('issues', []))} issues"
    if node_name == "city_route_planner" and state.get("city_route_plan"):
        route = state["city_route_plan"]
        return f"{len(route.stays)} stays, {len(route.segments)} route segments"
    if node_name == "repair_strategy_planner" and state.get("repair_strategy"):
        strategy = state["repair_strategy"]
        return f"{strategy.action.value}: {strategy.reason}"
    if node_name == "final_writer":
        return "final plan rendered"
    return "completed"


def _extract_plan_transfers(plan: TripPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in plan.route_segments:
        rows.append(
            {
                "day": "",
                "date": segment.departure_date.isoformat() if segment.departure_date else "",
                "label": f"route_{segment.segment_type.value}",
                "origin": segment.origin,
                "destination": segment.destination,
                "mode": segment.mode.value,
                "duration_minutes": segment.estimated_duration_minutes,
                "distance_km": segment.estimated_distance_km,
                "cost": segment.estimated_cost,
                "notes": segment.notes or segment.booking_notes,
            }
        )
    for day in plan.days:
        for item in day.timeline:
            if not item.move:
                continue
            rows.append(
                {
                    "day": day.day,
                    "date": day.date.isoformat(),
                    "label": item.move.purpose.value,
                    "time": f"{item.start_time.strftime('%H:%M')}-{item.end_time.strftime('%H:%M')}",
                    "origin": item.move.origin,
                    "destination": item.move.destination,
                    "mode": item.move.mode.value,
                    "duration_minutes": item.move.duration_minutes or item.duration_minutes,
                    "distance_km": item.move.distance_km,
                    "cost": item.move.estimated_cost,
                    "notes": item.move.notes,
                }
            )
    return rows


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _mcp_lodging(mcp_results: Any) -> list[Any]:
    value = getattr(mcp_results, "lodging", None)
    if value is None:
        if isinstance(mcp_results, dict):
            raw_value = mcp_results.get("lodging", [])
            return raw_value if isinstance(raw_value, list) else []
        return []
    return value if isinstance(value, list) else []


def _parse_text_list(value: str) -> list[str]:
    normalized = value.replace("\n", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _clear_stale_session_state() -> None:
    if st.session_state.get("ui_state_version") == UI_STATE_VERSION:
        return
    for key in ("last_result", "last_events", "last_log_path"):
        st.session_state.pop(key, None)
    st.session_state["ui_state_version"] = UI_STATE_VERSION


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            border: 1px solid #d6d9de;
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            background: #ffffff;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
