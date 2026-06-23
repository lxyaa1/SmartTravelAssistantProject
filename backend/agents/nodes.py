from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from typing import Any

from backend.agents.llm import (
    generate_city_route_plan_with_llm,
    generate_initial_plan_with_llm,
    replan_with_llm,
    should_use_llm,
)
from backend.graph.state import TripState
from backend.mcp.amap import execute_amap_mcp_query_plan, should_use_amap_mcp
from backend.schemas.trip import (
    AccommodationAreaResult,
    AccommodationStay,
    AttractionResult,
    BudgetLevel,
    CityRoutePlan,
    CityRoutePlannerInput,
    CityRoutePlannerOutput,
    CityStayPlan,
    DataCollectorInput,
    DataCollectorOutput,
    DraftDayScheduleInput,
    DraftDayScheduleOutput,
    FinalPlan,
    FinalWriterInput,
    FinalWriterOutput,
    IssueType,
    LodgingResult,
    McpQuery,
    McpQueryPlan,
    McpQueryStage,
    McpResults,
    McpToolName,
    MoveDetail,
    MovePurpose,
    ParsedRequestOutput,
    PlanCheckQueryPlannerInput,
    PlanCheckQueryPlannerOutput,
    PlanDay,
    PlanQualityGate,
    PlaceCategory,
    PreplanQueryPlannerInput,
    PreplanQueryPlannerOutput,
    RepairAction,
    RepairStrategy,
    RepairStrategyPlannerInput,
    RepairStrategyPlannerOutput,
    ReplannerInput,
    ReplannerOutput,
    RouteResult,
    SegmentType,
    Severity,
    StayDetail,
    StayPurpose,
    TimelineItem,
    TimelineItemType,
    TransportMode,
    TripPlan,
    TripRequest,
    TripSegment,
    ValidationIssue,
    ValidatorInput,
    ValidatorOutput,
    WeatherResult,
)
from mcp_servers.mock_travel_server.server import (
    get_attraction_detail as mock_get_attraction_detail,
    get_route_time as mock_get_route_time,
    get_weather as mock_get_weather,
    search_accommodation_areas as mock_search_accommodation_areas,
    search_attractions as mock_search_attractions,
)


def parse_request_node(state: TripState) -> TripState:
    raw = state.get("raw_user_input", {})
    travelers = raw.get("travelers")
    if travelers is None:
        travelers = {
            "adults": raw.get("adults", 1),
            "children": raw.get("children", 0),
            "infants": raw.get("infants", 0),
            "children_need_bed": raw.get("children_need_bed", 0),
            "infants_need_bed": raw.get("infants_need_bed", 0),
            "children_ages": raw.get("children_ages", []),
            "infants_ages": raw.get("infants_ages", []),
        }

    request = TripRequest(
        origin=raw.get("origin", "Shanghai"),
        destination=raw.get("destination", "Hangzhou"),
        start_date=_parse_date(raw.get("start_date", "2026-07-01")),
        end_date=_parse_date(raw.get("end_date", "2026-07-03")),
        travelers=travelers,
        accommodation=raw.get("accommodation"),
        budget_level=BudgetLevel(raw.get("budget_level", BudgetLevel.MEDIUM.value)),
        preferences=raw.get("preferences", ["culture", "food"]),
        must_visit=raw.get("must_visit", []),
        avoid=raw.get("avoid", []),
    )
    output = ParsedRequestOutput(request=request)
    return {
        **state,
        "user_request": output.request,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 3),
        "plan_versions": state.get("plan_versions", []),
        "pending_mcp_queries": McpQueryPlan(),
        "mcp_results": state.get("mcp_results", McpResults()),
        "mcp_errors": state.get("mcp_errors", []),
        "mcp_cache": state.get("mcp_cache", {}),
        "mcp_cache_stats": state.get("mcp_cache_stats", {"hits": 0, "misses": 0, "entries": 0}),
        "issues": [],
    }


def city_route_planner_node(state: TripState) -> TripState:
    agent_input = CityRoutePlannerInput(request=state["user_request"])
    if should_use_llm(state):
        city_route_plan = generate_city_route_plan_with_llm(request=agent_input.request)
    else:
        city_route_plan = _build_city_route_plan(agent_input.request)
    city_route_plan = _normalize_city_route_plan(city_route_plan, agent_input.request)
    output = CityRoutePlannerOutput(city_route_plan=city_route_plan)
    return {**state, "city_route_plan": output.city_route_plan}


def preplan_query_planner_node(state: TripState) -> TripState:
    agent_input = PreplanQueryPlannerInput(
        request=state["user_request"],
        city_route_plan=state.get("city_route_plan"),
    )
    request = agent_input.request
    city_route_plan = agent_input.city_route_plan or _build_city_route_plan(request)
    queries: list[McpQuery] = []

    for stay in city_route_plan.stays:
        queries.append(
            McpQuery(
                tool_name=McpToolName.SEARCH_ATTRACTIONS,
                args={"city": stay.city, "preferences": request.preferences},
                purpose="Find optional attractions and rainy-day backups for the city stay.",
                stage=McpQueryStage.PREPLAN,
            )
        )
        queries.append(
            McpQuery(
                tool_name=McpToolName.SEARCH_ACCOMMODATION_AREAS,
                args={
                    "city": stay.city,
                    "budget_level": request.budget_level.value,
                    "prefer_family_room": request.accommodation.prefer_family_room if request.accommodation else False,
                },
                purpose="Find accommodation areas for the city stay.",
                stage=McpQueryStage.PREPLAN,
            )
        )
        anchor = stay.lodging_anchor or (stay.anchor_places[0] if stay.anchor_places else stay.city)
        queries.append(
            McpQuery(
                tool_name=McpToolName.SEARCH_LODGING_NEAR_PLACE,
                args={
                    "city": stay.city,
                    "anchor_place": anchor,
                    "budget_level": request.budget_level.value,
                    "prefer_family_room": request.accommodation.prefer_family_room if request.accommodation else False,
                    "radius_km": 5,
                },
                purpose="Find lodging close to the primary day anchor.",
                stage=McpQueryStage.PREPLAN,
            )
        )
        for current_date in _date_range(stay.start_date, stay.end_date):
            queries.append(
                McpQuery(
                    tool_name=McpToolName.GET_WEATHER,
                    args={"city": stay.city, "date": current_date.isoformat()},
                    purpose="Check weather before assigning outdoor visits.",
                    stage=McpQueryStage.PREPLAN,
                )
            )

    for place in request.must_visit:
        target_stay = _stay_for_place(city_route_plan, place) or (city_route_plan.stays[0] if city_route_plan.stays else None)
        queries.append(
            McpQuery(
                tool_name=McpToolName.GET_ATTRACTION_DETAIL,
                args={
                    "name": place,
                    "city": target_stay.city if target_stay else request.destination,
                    "date": target_stay.start_date.isoformat() if target_stay else request.start_date.isoformat(),
                },
                purpose="Check must-visit availability and category.",
                stage=McpQueryStage.PREPLAN,
            )
        )

    for segment in city_route_plan.segments:
        queries.append(_route_query_from_segment(segment, McpQueryStage.PREPLAN))

    existing_results = state.get("mcp_results", McpResults())
    deduped_queries = _dedupe_mcp_queries(queries)
    filtered_queries = _filter_queries_with_existing_results(deduped_queries, existing_results)
    output = PreplanQueryPlannerOutput(query_plan=McpQueryPlan(queries=filtered_queries))
    return {
        **state,
        "pending_mcp_queries": output.query_plan,
        "mcp_cache_stats": _update_skipped_existing_stats(state, len(deduped_queries) - len(filtered_queries)),
    }


def collect_mcp_data_node(state: TripState) -> TripState:
    query_plan = state.get("pending_mcp_queries", McpQueryPlan())
    cache: dict[str, McpResults] = dict(state.get("mcp_cache", {}))
    cached_results = McpResults()
    missing_queries: list[McpQuery] = []
    cache_hits = 0
    cache_misses = 0

    for query in query_plan.queries:
        cache_key = _mcp_query_cache_key(query)
        cached = cache.get(cache_key)
        if cached is None:
            missing_queries.append(query)
            cache_misses += 1
        else:
            cached_results = _merge_mcp_results(cached_results, cached)
            cache_hits += 1

    agent_input = DataCollectorInput(
        query_plan=McpQueryPlan(queries=missing_queries),
        existing_results=state.get("mcp_results", McpResults()),
        default_city=state["user_request"].destination,
    )
    errors = list(state.get("mcp_errors", []))

    if not missing_queries:
        collected = McpResults()
    elif should_use_amap_mcp(state):
        try:
            collected = execute_amap_mcp_query_plan(
                query_plan=agent_input.query_plan,
                default_city=agent_input.default_city,
            )
        except Exception as exc:
            errors.append(
                "Amap MCP failed; live MCP batch was skipped and no mock data was mixed in: "
                f"{_format_exception_detail(exc)}"
            )
            collected = McpResults()
    else:
        collected = _execute_mock_mcp_query_plan(agent_input.query_plan, agent_input.default_city)

    errors.extend(collected.errors)
    for query in missing_queries:
        query_result = _extract_query_result_from_results(query, collected)
        if query_result is not None:
            cache[_mcp_query_cache_key(query)] = query_result

    merged_results = _merge_mcp_results(agent_input.existing_results, cached_results)
    merged_results = _merge_mcp_results(merged_results, collected)
    previous_stats = state.get("mcp_cache_stats", {})
    cache_stats = {
        "hits": int(previous_stats.get("hits", 0)) + cache_hits,
        "misses": int(previous_stats.get("misses", 0)) + cache_misses,
        "last_hits": cache_hits,
        "last_misses": cache_misses,
        "entries": len(cache),
        "skipped_existing_results": int(previous_stats.get("skipped_existing_results", 0)),
        "last_skipped_existing_results": int(previous_stats.get("last_skipped_existing_results", 0)),
    }
    output = DataCollectorOutput(mcp_results=merged_results)
    return {
        **state,
        "mcp_results": output.mcp_results,
        "pending_mcp_queries": McpQueryPlan(),
        "mcp_errors": errors,
        "mcp_cache": cache,
        "mcp_cache_stats": cache_stats,
    }


def draft_day_schedule_node(state: TripState) -> TripState:
    city_route_plan = state.get("city_route_plan") or _build_city_route_plan(state["user_request"])
    agent_input = DraftDayScheduleInput(
        request=state["user_request"],
        city_route_plan=city_route_plan,
        mcp_results=state.get("mcp_results", McpResults()),
    )
    if should_use_llm(state):
        plan = generate_initial_plan_with_llm(
            request=agent_input.request,
            mcp_results=agent_input.mcp_results,
            city_route_plan=agent_input.city_route_plan,
        )
    else:
        plan = _build_deterministic_trip_plan(
            request=agent_input.request,
            city_route_plan=agent_input.city_route_plan,
            mcp_results=agent_input.mcp_results,
        )
    _normalize_plan_after_generation(plan, agent_input.request, agent_input.city_route_plan, agent_input.mcp_results)
    output = DraftDayScheduleOutput(plan=plan)
    return {**state, "current_plan": output.plan}


def initial_plan_node(state: TripState) -> TripState:
    return draft_day_schedule_node(state)


def plan_check_query_planner_node(state: TripState) -> TripState:
    agent_input = PlanCheckQueryPlannerInput(plan=state["current_plan"])
    plan = agent_input.plan
    queries: list[McpQuery] = []
    accommodation_locations = _accommodation_locations_by_name(plan)

    for segment in plan.route_segments:
        queries.append(_route_query_from_segment(segment, McpQueryStage.PLAN_CHECK))

    for day in plan.days:
        queries.append(
            McpQuery(
                tool_name=McpToolName.GET_WEATHER,
                args={"city": day.city, "date": day.date.isoformat()},
                purpose="Verify weather for the planned day.",
                stage=McpQueryStage.PLAN_CHECK,
            )
        )
        for item in day.timeline:
            if item.item_type == TimelineItemType.MOVE and item.move:
                if _move_has_unverified_endpoint(item.move):
                    continue
                route_query = _route_query_from_move(item.move, McpQueryStage.PLAN_CHECK)
                _attach_known_endpoint_locations(route_query, accommodation_locations)
                queries.append(route_query)
            if item.item_type == TimelineItemType.STAY and item.stay and item.stay.purpose == StayPurpose.VISIT:
                queries.append(
                    McpQuery(
                        tool_name=McpToolName.GET_ATTRACTION_DETAIL,
                        args={"name": item.stay.place_name, "city": item.stay.city or day.city, "date": day.date.isoformat()},
                        purpose="Verify attraction status for the planned date.",
                        stage=McpQueryStage.PLAN_CHECK,
                    )
                )

    existing_results = state.get("mcp_results", McpResults())
    deduped_queries = _dedupe_mcp_queries(queries)
    filtered_queries = _filter_queries_with_existing_results(deduped_queries, existing_results)
    output = PlanCheckQueryPlannerOutput(query_plan=McpQueryPlan(queries=filtered_queries))
    return {
        **state,
        "pending_mcp_queries": output.query_plan,
        "mcp_cache_stats": _update_skipped_existing_stats(state, len(deduped_queries) - len(filtered_queries)),
    }


def validate_plan_node(state: TripState) -> TripState:
    agent_input = ValidatorInput(
        request=state["user_request"],
        plan=state["current_plan"],
        mcp_results=state.get("mcp_results", McpResults()),
    )
    request = agent_input.request
    plan = agent_input.plan
    mcp_results = agent_input.mcp_results
    route_by_key = {_route_result_key(route): route for route in mcp_results.routes}
    _apply_route_results_to_plan(plan, route_by_key)
    issues = _validate_plan(request, plan, mcp_results)
    quality_gate = _quality_gate_for_issues(issues)
    plan.quality_gate = quality_gate
    output = ValidatorOutput(issues=issues, quality_gate=quality_gate)
    return {**state, "current_plan": plan, "issues": output.issues}


def repair_strategy_planner_node(state: TripState) -> TripState:
    agent_input = RepairStrategyPlannerInput(
        issues=state.get("issues", []),
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 3),
    )
    serious = [issue for issue in agent_input.issues if issue.severity in {Severity.HIGH, Severity.CRITICAL}]
    missing_external_data = [issue for issue in serious if issue.issue_type == IssueType.MISSING_MCP_DATA]
    if not serious:
        action = RepairAction.FINALIZE
        reason = "No high or critical issues remain."
    elif missing_external_data:
        action = RepairAction.FINALIZE
        reason = "Required MCP data is missing; do not replan from guessed or mock data."
    elif agent_input.iteration >= agent_input.max_iterations:
        action = RepairAction.FINALIZE
        reason = "Maximum replanning iterations reached; final output is provisional."
    else:
        action = RepairAction.REPLAN
        reason = "High or critical issues remain and another replanning iteration is available."
    output = RepairStrategyPlannerOutput(
        repair_strategy=RepairStrategy(
            action=action,
            reason=reason,
            target_issue_types=[issue.issue_type for issue in serious],
        )
    )
    return {**state, "repair_strategy": output.repair_strategy}


def replan_node(state: TripState) -> TripState:
    agent_input = ReplannerInput(
        request=state["user_request"],
        current_plan=state["current_plan"],
        issues=state.get("issues", []),
        mcp_results=state.get("mcp_results", McpResults()),
        iteration=state.get("iteration", 0),
    )
    previous_versions = list(state.get("plan_versions", []))
    previous_versions.append(agent_input.current_plan.model_copy(deep=True))

    if should_use_llm(state):
        plan = replan_with_llm(
            request=agent_input.request,
            current_plan=agent_input.current_plan,
            issues=agent_input.issues,
            mcp_results=agent_input.mcp_results,
        )
    else:
        plan = _deterministic_replan(agent_input)
    _normalize_plan_after_generation(
        plan,
        agent_input.request,
        state.get("city_route_plan") or _build_city_route_plan(agent_input.request),
        agent_input.mcp_results,
    )
    output = ReplannerOutput(plan=plan, addressed_issues=agent_input.issues)
    return {
        **state,
        "current_plan": output.plan,
        "plan_versions": previous_versions,
        "iteration": agent_input.iteration + 1,
        "issues": [],
        "repair_strategy": None,
    }


def final_writer_node(state: TripState) -> TripState:
    agent_input = FinalWriterInput(
        plan=state["current_plan"],
        unresolved_issues=state.get("issues", []),
    )
    plan = agent_input.plan
    lines = [f"# {plan.title}", "", f"{plan.origin} -> {plan.destination}", ""]
    if not plan.quality_gate.can_finalize:
        lines.extend(
            [
                "## 状态",
                "",
                "当前计划还不能作为可执行行程使用，因为仍有必须解决的校验问题。",
                plan.quality_gate.reason,
                "",
            ]
        )

    if plan.route_segments:
        lines.extend(["## Route Skeleton", ""])
        for segment in plan.route_segments:
            lines.append(
                "- "
                f"{segment.segment_type.value}: {segment.origin} -> {segment.destination} "
                f"by {segment.mode.value}, {_format_date_time(segment.departure_date, segment.departure_time)}"
                f" to {_format_date_time(segment.arrival_date, segment.arrival_time)}, "
                f"{segment.estimated_duration_minutes} min"
            )
        lines.append("")

    if plan.accommodations:
        lines.extend(["## Accommodation", ""])
        for stay in plan.accommodations:
            lines.append(
                f"- {stay.city}: {stay.hotel_name}, {stay.check_in_date} to {stay.check_out_date}, "
                f"near {', '.join(stay.nearby_anchor_places) or stay.area}"
            )
        lines.append("")

    for day in plan.days:
        lines.append(f"## Day {day.day} - {day.date} - {day.city}")
        if day.accommodation_area:
            lines.append(f"Accommodation area: {day.accommodation_area}")
        for item in sorted(day.timeline, key=lambda current: current.sequence):
            time_label = f"{item.start_time.strftime('%H:%M')}-{item.end_time.strftime('%H:%M')}"
            if item.item_type == TimelineItemType.MOVE and item.move:
                lines.append(
                    f"- {time_label} MOVE {item.move.origin} -> {item.move.destination} "
                    f"by {item.move.mode.value}, {item.move.duration_minutes or item.duration_minutes} min"
                )
            elif item.stay:
                lines.append(
                    f"- {time_label} STAY {item.stay.place_name} "
                    f"({item.stay.purpose.value}), {item.stay.duration_minutes or item.duration_minutes} min"
                )
        if day.daily_notes:
            lines.append(f"Notes: {day.daily_notes}")
        lines.append("")

    if agent_input.unresolved_issues:
        lines.extend(["## Unresolved Issues", ""])
        for issue in agent_input.unresolved_issues:
            lines.append(f"- {issue.severity.value}: {issue.reason} Suggested: {issue.suggested_action}")

    output = FinalWriterOutput(final_plan=FinalPlan(content="\n".join(lines), unresolved_issues=agent_input.unresolved_issues))
    return {**state, "final_plan": output.final_plan}


def _build_city_route_plan(request: TripRequest) -> CityRoutePlan:
    days = _date_range(request.start_date, request.end_date)
    anchors = request.must_visit or [request.destination]
    grouped: dict[str, list[str]] = {}
    for place in anchors:
        grouped.setdefault(_anchor_city_for_place(request.destination, place), []).append(place)
    if not grouped:
        grouped[request.destination] = []

    cities = list(grouped.keys())
    stays: list[CityStayPlan] = []
    date_index = 0
    for index, city in enumerate(cities, start=1):
        remaining_cities = len(cities) - index + 1
        remaining_days = len(days) - date_index
        stay_length = max(1, remaining_days - remaining_cities + 1)
        start = days[date_index]
        end = days[min(len(days) - 1, date_index + stay_length - 1)]
        anchors_for_city = grouped[city]
        stays.append(
            CityStayPlan(
                sequence=index,
                city=city,
                start_date=start,
                end_date=end,
                anchor_places=anchors_for_city,
                lodging_anchor=anchors_for_city[0] if anchors_for_city else city,
                notes="Deterministic city stay inferred from destination and must-visit places.",
            )
        )
        date_index += stay_length

    segments: list[TripSegment] = []
    sequence = 1
    first_city = stays[0].city
    if request.origin != first_city:
        segments.append(
            TripSegment(
                sequence=sequence,
                segment_type=SegmentType.OUTBOUND,
                origin=request.origin,
                destination=first_city,
                origin_city=request.origin,
                destination_city=first_city,
                mode=TransportMode.TRAIN,
                departure_date=request.start_date,
                departure_time=time(8, 0),
                arrival_date=request.start_date,
                arrival_time=time(12, 0),
                estimated_duration_minutes=240,
                estimated_cost=180,
                booking_notes="Train number is not available from current MCP tools.",
            )
        )
        sequence += 1

    for previous, current in zip(stays, stays[1:]):
        segments.append(
            TripSegment(
                sequence=sequence,
                segment_type=SegmentType.INTERCITY,
                origin=previous.city,
                destination=current.city,
                origin_city=previous.city,
                destination_city=current.city,
                mode=TransportMode.TRAIN,
                departure_date=current.start_date,
                departure_time=time(8, 30),
                arrival_date=current.start_date,
                arrival_time=time(11, 30),
                estimated_duration_minutes=180,
                estimated_cost=120,
                booking_notes="Intercity rail detail needs a ticket source.",
            )
        )
        sequence += 1

    last_city = stays[-1].city
    if last_city != request.origin:
        segments.append(
            TripSegment(
                sequence=sequence,
                segment_type=SegmentType.RETURN,
                origin=last_city,
                destination=request.origin,
                origin_city=last_city,
                destination_city=request.origin,
                mode=TransportMode.TRAIN,
                departure_date=request.end_date,
                departure_time=time(17, 0),
                arrival_date=request.end_date,
                arrival_time=time(21, 0),
                estimated_duration_minutes=240,
                estimated_cost=180,
                booking_notes="Return train number is not available from current MCP tools.",
            )
        )

    return CityRoutePlan(origin=request.origin, destination=request.destination, stays=stays, segments=segments)


def _normalize_city_route_plan(plan: CityRoutePlan, request: TripRequest) -> CityRoutePlan:
    if not plan.stays:
        return _build_city_route_plan(request)
    plan.origin = plan.origin or request.origin
    plan.destination = plan.destination or request.destination
    for index, stay in enumerate(plan.stays, start=1):
        stay.sequence = index
        if not stay.anchor_places:
            stay.anchor_places = [
                place
                for place in request.must_visit
                if _anchor_city_for_place(request.destination, place) == stay.city
            ]
        if not stay.lodging_anchor:
            stay.lodging_anchor = stay.anchor_places[0] if stay.anchor_places else stay.city
    if not any(segment.segment_type == SegmentType.RETURN for segment in plan.segments) and plan.stays[-1].city != request.origin:
        fallback = _build_city_route_plan(request)
        plan.segments.extend([segment for segment in fallback.segments if segment.segment_type == SegmentType.RETURN])
    for index, segment in enumerate(plan.segments, start=1):
        segment.sequence = index
        _fill_segment_city_context(segment, plan, request)
    return plan


def _build_deterministic_trip_plan(
    request: TripRequest,
    city_route_plan: CityRoutePlan,
    mcp_results: McpResults,
) -> TripPlan:
    route_segments = [segment.model_copy(deep=True) for segment in city_route_plan.segments]
    _apply_route_results_to_segments(route_segments, mcp_results)
    accommodations = _build_accommodations(request, city_route_plan, mcp_results)
    accommodation_by_city = {stay.city: stay for stay in accommodations}
    days: list[PlanDay] = []
    for current_date in _date_range(request.start_date, request.end_date):
        stay = _city_stay_for_date(city_route_plan, current_date)
        city = stay.city if stay else request.destination
        accommodation = accommodation_by_city.get(city)
        timeline = _build_day_timeline(
            request=request,
            city_route_plan=city_route_plan,
            mcp_results=mcp_results,
            current_date=current_date,
            city=city,
            accommodation=accommodation,
            route_segments=route_segments,
        )
        day = PlanDay(
            day=(current_date - request.start_date).days + 1,
            date=current_date,
            city=city,
            timeline=timeline,
            accommodation_area=accommodation.hotel_name if accommodation else _preferred_accommodation_area(mcp_results, city),
            overnight_accommodation=accommodation.hotel_name if accommodation and current_date < request.end_date else None,
            daily_notes="Initial deterministic draft. MCP validation may adjust move durations.",
        )
        _recalculate_day_totals(day)
        days.append(day)

    plan = TripPlan(
        title=f"{request.origin} to {request.destination} travel plan",
        origin=request.origin,
        destination=request.destination,
        route_segments=route_segments,
        accommodations=accommodations,
        days=days,
        assumptions=[
            "Train and flight numbers require a separate ticket source; current MCP tools only provide map and POI data.",
            "Each daily timeline is represented as move/stay items.",
        ],
    )
    _recalculate_plan_totals(plan)
    return plan


def _build_accommodations(
    request: TripRequest,
    city_route_plan: CityRoutePlan,
    mcp_results: McpResults,
) -> list[AccommodationStay]:
    accommodations: list[AccommodationStay] = []
    for stay in city_route_plan.stays:
        lodging = _best_lodging_for_anchor(mcp_results, stay.city, stay.lodging_anchor)
        area = lodging.area if lodging else _preferred_accommodation_area(mcp_results, stay.city, stay.lodging_anchor)
        name = lodging.name if lodging else area
        accommodations.append(
            AccommodationStay(
                hotel_name=name,
                city=stay.city,
                area=area,
                address=lodging.address if lodging else "",
                location=lodging.location if lodging else "",
                check_in_date=stay.start_date,
                check_out_date=stay.end_date + timedelta(days=1),
                bed_count=request.accommodation.bed_count if request.accommodation else request.travelers.bed_count,
                room_count=request.accommodation.room_count if request.accommodation else None,
                reason=f"Near {stay.lodging_anchor or stay.city}.",
                nearby_anchor_places=stay.anchor_places or ([stay.lodging_anchor] if stay.lodging_anchor else []),
                estimated_cost_per_night=_cost_for_budget(request.budget_level, low=260, medium=520, high=900),
            )
        )
    return accommodations


def _build_day_timeline(
    request: TripRequest,
    city_route_plan: CityRoutePlan,
    mcp_results: McpResults,
    current_date: date,
    city: str,
    accommodation: AccommodationStay | None,
    route_segments: list[TripSegment],
) -> list[TimelineItem]:
    items: list[TimelineItem] = []
    cursor = 0
    lodging_is_unverified = bool(accommodation and _is_unverified_lodging_label(accommodation.hotel_name))

    def add_stay(
        start_minute: int,
        end_minute: int,
        place_name: str,
        purpose: StayPurpose,
        category: PlaceCategory | None = None,
        activity: str = "",
        cost: float = 0,
        notes: str = "",
    ) -> None:
        if end_minute <= start_minute:
            return
        items.append(
            TimelineItem(
                sequence=len(items) + 1,
                item_type=TimelineItemType.STAY,
                start_time=_minutes_to_time(start_minute),
                end_time=_minutes_to_time(end_minute),
                city=city,
                stay=StayDetail(
                    place_name=place_name,
                    city=city,
                    purpose=purpose,
                    category=category,
                    activity=activity,
                    duration_minutes=end_minute - start_minute,
                    estimated_cost=cost,
                    notes=notes,
                ),
            )
        )

    def add_move(
        start_minute: int,
        duration_minutes: int,
        origin: str,
        destination: str,
        mode: TransportMode,
        purpose: MovePurpose,
        cost: float = 0,
        distance_km: float = 0,
        origin_city: str | None = None,
        destination_city: str | None = None,
        notes: str = "",
    ) -> int:
        if start_minute >= 23 * 60 + 59:
            return 23 * 60 + 59
        duration = max(1, duration_minutes)
        end_minute = min(23 * 60 + 59, start_minute + duration)
        if end_minute <= start_minute:
            return start_minute
        items.append(
            TimelineItem(
                sequence=len(items) + 1,
                item_type=TimelineItemType.MOVE,
                start_time=_minutes_to_time(start_minute),
                end_time=_minutes_to_time(end_minute),
                city=city,
                move=MoveDetail(
                    origin=origin,
                    destination=destination,
                    origin_city=origin_city or city,
                    destination_city=destination_city or city,
                    mode=mode,
                    purpose=purpose,
                    duration_minutes=end_minute - start_minute,
                    distance_km=distance_km,
                    estimated_cost=cost,
                    notes=notes,
                ),
            )
        )
        return end_minute

    add_stay(0, 7 * 60 + 30, accommodation.hotel_name if accommodation else city, StayPurpose.SLEEP, notes="Overnight rest.")
    add_stay(7 * 60 + 30, 8 * 60 + 15, "Breakfast", StayPurpose.MEAL, PlaceCategory.FOOD)
    cursor = 8 * 60 + 15

    for segment in _segments_for_date(route_segments, current_date):
        start = max(cursor, _time_to_minutes(segment.departure_time) if segment.departure_time else cursor)
        purpose = _move_purpose_from_segment(segment.segment_type)
        if segment.segment_type == SegmentType.RETURN and accommodation and not _locations_equivalent(
            segment.origin,
            accommodation.hotel_name,
        ):
            transfer_start = min(cursor, max(cursor, start - 25))
            cursor = add_move(
                transfer_start,
                _route_duration_for(mcp_results, accommodation.hotel_name, segment.origin, TransportMode.TAXI, fallback=25),
                accommodation.hotel_name,
                segment.origin,
                TransportMode.TAXI,
                MovePurpose.LOCAL,
                cost=35,
                origin_city=city,
                destination_city=segment.origin_city or city,
                notes="Lodging to return departure point.",
            )
            start = max(cursor, start)
        cursor = add_move(
            start,
            segment.estimated_duration_minutes or 180,
            segment.origin,
            segment.destination,
            segment.mode,
            purpose,
            cost=segment.estimated_cost,
            distance_km=segment.estimated_distance_km,
            origin_city=segment.origin_city or segment.origin,
            destination_city=segment.destination_city or segment.destination,
            notes=segment.booking_notes or segment.notes,
        )
        if segment.segment_type == SegmentType.RETURN:
            add_stay(cursor, min(23 * 60 + 59, cursor + 60), request.origin, StayPurpose.REST, notes="Arrival buffer.")
            cursor = min(23 * 60 + 59, cursor + 60)
            break
        add_stay(cursor, min(14 * 60, cursor + 45), "Meal or arrival buffer", StayPurpose.MEAL, PlaceCategory.FOOD)
        cursor = max(cursor + 45, 13 * 60)
        if accommodation and not lodging_is_unverified:
            if not _locations_equivalent(accommodation.hotel_name, segment.destination):
                cursor = add_move(
                    cursor,
                    _route_duration_for(mcp_results, segment.destination, accommodation.hotel_name, TransportMode.TAXI, fallback=25),
                    segment.destination,
                    accommodation.hotel_name,
                    TransportMode.TAXI,
                    MovePurpose.LOCAL,
                    cost=35,
                    origin_city=segment.destination_city or city,
                    destination_city=city,
                    notes="Arrival point to lodging.",
                )
            add_stay(cursor, min(cursor + 30, 14 * 60), accommodation.hotel_name, StayPurpose.HOTEL_CHECKIN)
            cursor = max(cursor + 30, 14 * 60)

    if not any(item.move and item.move.purpose == MovePurpose.RETURN for item in items):
        visit_candidates = _visits_for_date(request, city_route_plan, mcp_results, current_date, city)
        lodging_name = accommodation.hotel_name if accommodation and not lodging_is_unverified else city
        for index, place in enumerate(visit_candidates):
            detail = _attraction_detail_for(mcp_results, place, city, current_date)
            category = detail.category if detail else _category_for_place(place, mcp_results, PlaceCategory.OUTDOOR)
            recommended = detail.recommended_duration_minutes if detail else 120
            if cursor < 9 * 60:
                cursor = 9 * 60
            if index == 0:
                travel_minutes = _route_duration_for(mcp_results, lodging_name, place, TransportMode.TAXI, fallback=35)
            elif items and items[-1].stay:
                travel_minutes = _route_duration_for(
                    mcp_results,
                    items[-1].stay.place_name,
                    place,
                    TransportMode.TAXI,
                    fallback=35,
                )
            else:
                travel_minutes = 0
            if cursor + travel_minutes + min(recommended, 45) > 18 * 60:
                break
            if index == 0:
                cursor = add_move(
                    cursor,
                    travel_minutes,
                    lodging_name,
                    place,
                    TransportMode.TAXI,
                    MovePurpose.LOCAL,
                    cost=35,
                    notes="Hotel to first planned stay.",
                )
            elif items and items[-1].stay:
                previous_place = items[-1].stay.place_name
                cursor = add_move(
                    cursor,
                    travel_minutes,
                    previous_place,
                    place,
                    TransportMode.TAXI,
                    MovePurpose.LOCAL,
                    cost=30,
                    notes="Move between planned stays.",
                )
            visit_end = min(cursor + recommended, 17 * 60 + 30)
            add_stay(
                cursor,
                visit_end,
                place,
                StayPurpose.VISIT,
                category=category,
                activity="Visit attraction",
                cost=detail.ticket_price if detail else 30,
                notes=detail.notes if detail else "",
            )
            cursor = visit_end
            if cursor >= 17 * 60:
                break

        if (
            items
            and items[-1].stay
            and items[-1].stay.purpose == StayPurpose.VISIT
            and current_date < request.end_date
            and not lodging_is_unverified
        ):
            cursor = add_move(
                cursor,
                _route_duration_for(mcp_results, items[-1].stay.place_name, lodging_name, TransportMode.TAXI, fallback=35),
                items[-1].stay.place_name,
                lodging_name,
                TransportMode.TAXI,
                MovePurpose.LOCAL,
                cost=35,
                notes="Return to lodging.",
            )

        if cursor < 18 * 60 + 30:
            add_stay(cursor, 18 * 60 + 30, "Rest buffer", StayPurpose.REST)
            cursor = 18 * 60 + 30
        add_stay(cursor, min(cursor + 60, 20 * 60), "Dinner", StayPurpose.MEAL, PlaceCategory.FOOD)
        cursor = min(cursor + 60, 20 * 60)
        add_stay(cursor, 22 * 60 + 30, accommodation.hotel_name if accommodation else city, StayPurpose.REST)
        cursor = 22 * 60 + 30
        add_stay(cursor, 23 * 60 + 59, accommodation.hotel_name if accommodation else city, StayPurpose.SLEEP)

    return _dedupe_and_sort_timeline(items)


def _visits_for_date(
    request: TripRequest,
    city_route_plan: CityRoutePlan,
    mcp_results: McpResults,
    current_date: date,
    city: str,
) -> list[str]:
    day_number = (current_date - request.start_date).days + 1
    city_stay = _city_stay_for_date(city_route_plan, current_date)
    city_anchors = list(city_stay.anchor_places if city_stay else [])
    if not city_anchors:
        city_anchors = [place for place in request.must_visit if _anchor_city_for_place(request.destination, place) == city]

    visits: list[str] = []
    if city_anchors:
        # The first draft intentionally keeps the first must-visit early so validation can catch weather/route problems.
        visits.append(city_anchors[0])

    optional = [
        attraction.name
        for attraction in mcp_results.attractions
        if attraction.city == city
        and attraction.date is None
        and attraction.name not in visits
        and _attraction_matches_city_anchor(attraction.name, city_anchors)
    ]
    visits.extend(optional[:2])

    if not visits:
        visits = [f"{city} Main Attraction", f"{city} Old Street"]
    if day_number > 1 and city_anchors:
        visits = city_anchors + [item for item in optional if item not in city_anchors]
    return visits[:2]


def _normalize_plan_after_generation(
    plan: TripPlan,
    request: TripRequest,
    city_route_plan: CityRoutePlan,
    mcp_results: McpResults,
) -> None:
    if not plan.route_segments:
        plan.route_segments = [segment.model_copy(deep=True) for segment in city_route_plan.segments]
    for index, segment in enumerate(plan.route_segments, start=1):
        segment.sequence = index
        _fill_segment_city_context(segment, city_route_plan, request)
    _apply_route_results_to_segments(plan.route_segments, mcp_results)

    if not plan.accommodations:
        plan.accommodations = _build_accommodations(request, city_route_plan, mcp_results)
    accommodation_by_city = {stay.city: stay for stay in plan.accommodations}

    dates = _date_range(request.start_date, request.end_date)
    if not plan.days:
        for current_date in dates:
            stay = _city_stay_for_date(city_route_plan, current_date)
            city = stay.city if stay else request.destination
            accommodation = accommodation_by_city.get(city)
            plan.days.append(
                PlanDay(
                    day=(current_date - request.start_date).days + 1,
                    date=current_date,
                    city=city,
                    timeline=_build_day_timeline(
                        request,
                        city_route_plan,
                        mcp_results,
                        current_date,
                        city,
                        accommodation,
                        plan.route_segments,
                    ),
                    accommodation_area=accommodation.hotel_name if accommodation else None,
                    overnight_accommodation=accommodation.hotel_name if accommodation and current_date < request.end_date else None,
                )
            )

    for index, day in enumerate(plan.days, start=1):
        day.day = index
        if not day.timeline:
            stay = _city_stay_for_date(city_route_plan, day.date)
            accommodation = accommodation_by_city.get(day.city)
            day.timeline = _build_day_timeline(
                request,
                city_route_plan,
                mcp_results,
                day.date,
                day.city or (stay.city if stay else request.destination),
                accommodation,
                plan.route_segments,
            )
        day.timeline = _dedupe_and_sort_timeline(day.timeline)
        _sync_local_move_endpoints(day)
        _fill_day_move_city_context(day, request, city_route_plan)
        _normalize_passive_boundary_stays(day, request, plan.days)
        _normalize_passive_stay_locations(day)
        _repair_day_timeline_continuity(day)
        _remove_unverified_lodging_moves(day)
        _remove_same_location_moves(day)
        _fit_day_timeline_sequentially(day)
        if not day.accommodation_area and day.city in accommodation_by_city:
            day.accommodation_area = accommodation_by_city[day.city].hotel_name
        _recalculate_day_totals(day)
    _repair_cross_day_timeline_continuity(plan)
    for day in plan.days:
        _remove_unverified_lodging_moves(day)
        _remove_same_location_moves(day)
        _fit_day_timeline_sequentially(day)
        _recalculate_day_totals(day)
    _recalculate_plan_totals(plan)


def _deterministic_replan(agent_input: ReplannerInput) -> TripPlan:
    plan = agent_input.current_plan.model_copy(deep=True)
    issue_types = {issue.issue_type for issue in agent_input.issues}

    if IssueType.BAD_WEATHER in issue_types:
        _move_rainy_outdoor_visits(plan, agent_input.request, agent_input.mcp_results)
    if IssueType.LODGING_TOO_FAR in issue_types or IssueType.ROUTE_TOO_LONG in issue_types:
        _repair_lodging_and_local_moves(plan, agent_input.mcp_results)
    if IssueType.MISSING_RETURN_TRANSFER in issue_types:
        _ensure_return_move(plan, agent_input.request)
    if IssueType.MISSING_MUST_VISIT in issue_types:
        _ensure_must_visits(plan, agent_input.request, agent_input.mcp_results)

    _recalculate_plan_totals(plan)
    plan.assumptions.append(f"Deterministic replan iteration {agent_input.iteration + 1} applied.")
    return plan


def _move_rainy_outdoor_visits(plan: TripPlan, request: TripRequest, mcp_results: McpResults) -> None:
    for day in plan.days:
        if not _is_bad_weather(mcp_results, day.city, day.date):
            continue
        for item in day.timeline:
            if not item.stay or item.stay.purpose != StayPurpose.VISIT or item.stay.category != PlaceCategory.OUTDOOR:
                continue
            if any(_place_matches_any(required, [item.stay.place_name]) for required in request.must_visit):
                item.stay.notes = "Must-visit outdoor stop retained despite bad weather; user should confirm risk."
                continue
            replacement = _select_indoor_attraction(mcp_results, day.city, excluded={item.stay.place_name})
            original_place = item.stay.place_name
            if replacement:
                item.stay.place_name = replacement.name
                item.stay.category = replacement.category
                item.stay.estimated_cost = replacement.ticket_price
                item.stay.notes = "Replaced outdoor visit because of bad weather."
            else:
                item.stay.place_name = day.accommodation_area or day.overnight_accommodation or day.city
                item.stay.purpose = StayPurpose.REST
                item.stay.category = PlaceCategory.HOTEL_AREA
                item.stay.estimated_cost = 0
                item.stay.notes = f"Removed {original_place} because of bad weather; no verified indoor replacement was available."
    for day in plan.days:
        _resequence_timeline(day.timeline)


def _repair_lodging_and_local_moves(plan: TripPlan, mcp_results: McpResults) -> None:
    for day in plan.days:
        first_visit = _first_visit(day)
        if first_visit is None:
            continue
        lodging = _best_lodging_for_anchor(mcp_results, day.city, first_visit.stay.place_name)
        if lodging is None:
            if _day_has_unreliable_lodging_routes(day):
                _mark_lodging_unresolved(day, first_visit.stay.place_name)
                _fit_day_timeline_sequentially(day)
                _recalculate_day_totals(day)
            continue
        previous_lodging = day.accommodation_area
        day.accommodation_area = lodging.name
        if day.overnight_accommodation:
            day.overnight_accommodation = lodging.name
        for item in day.timeline:
            if item.stay and item.stay.purpose in {StayPurpose.SLEEP, StayPurpose.REST, StayPurpose.HOTEL_CHECKIN, StayPurpose.HOTEL_CHECKOUT}:
                item.stay.place_name = lodging.name
            if item.move and item.move.purpose == MovePurpose.LOCAL:
                if _endpoint_is_lodging_reference(item.move.origin, previous_lodging) or _endpoint_is_lodging_reference(
                    item.move.origin,
                    day.accommodation_area,
                ):
                    item.move.origin = lodging.name
                if _endpoint_is_lodging_reference(item.move.destination, previous_lodging) or _endpoint_is_lodging_reference(
                    item.move.destination,
                    day.accommodation_area,
                ):
                    item.move.destination = lodging.name
                if _locations_equivalent(first_visit.stay.place_name, item.move.origin) or _locations_equivalent(
                    first_visit.stay.place_name,
                    item.move.destination,
                ):
                    item.move.duration_minutes = max(10, lodging.duration_to_anchor_minutes or 15)
                    item.move.distance_km = lodging.distance_to_anchor_km or 2
                    item.end_time = _shift_time(item.start_time, item.move.duration_minutes)
                    item.move.notes = "Adjusted to lodging near the anchor."
        _recalculate_day_totals(day)
    _sync_accommodations_with_days(plan, mcp_results)


def _ensure_return_move(plan: TripPlan, request: TripRequest) -> None:
    if request.origin == request.destination:
        return
    last_day = plan.days[-1]
    if any(item.move and item.move.purpose == MovePurpose.RETURN for item in last_day.timeline):
        return
    start = 17 * 60
    last_city = last_day.city
    last_day.timeline.append(
        TimelineItem(
            sequence=len(last_day.timeline) + 1,
            item_type=TimelineItemType.MOVE,
            start_time=_minutes_to_time(start),
            end_time=_minutes_to_time(min(23 * 60 + 59, start + 240)),
            city=last_day.city,
            move=MoveDetail(
                origin=last_city,
                destination=request.origin,
                origin_city=last_city,
                destination_city=request.origin,
                mode=TransportMode.TRAIN,
                purpose=MovePurpose.RETURN,
                duration_minutes=240,
                estimated_cost=180,
                notes="Added explicit return move during replan.",
            ),
        )
    )
    last_day.timeline = _dedupe_and_sort_timeline(last_day.timeline)


def _ensure_must_visits(plan: TripPlan, request: TripRequest, mcp_results: McpResults) -> None:
    planned = [item.stay.place_name for day in plan.days for item in day.timeline if item.stay and item.stay.purpose == StayPurpose.VISIT]
    for place in request.must_visit:
        if _place_matches_any(place, planned):
            continue
        target_day = _next_non_rain_day(plan, 0, _anchor_city_for_place(request.destination, place), mcp_results) or plan.days[-1]
        _replace_optional_visit(target_day, place, mcp_results)
        planned.append(place)


def _replace_optional_visit(day: PlanDay, place: str, mcp_results: McpResults) -> None:
    for item in reversed(day.timeline):
        if item.stay and item.stay.purpose == StayPurpose.VISIT:
            detail = _attraction_detail_for(mcp_results, place, day.city, day.date)
            item.stay.place_name = place
            item.stay.category = detail.category if detail else _category_for_place(place, mcp_results, PlaceCategory.OUTDOOR)
            item.stay.estimated_cost = detail.ticket_price if detail else item.stay.estimated_cost
            item.stay.notes = "Inserted must-visit during replan."
            return


def _attraction_matches_city_anchor(attraction_name: str, anchors: list[str]) -> bool:
    if not anchors:
        return True
    normalized_attraction = _normalize_place_name(attraction_name)
    normalized_anchors = [_normalize_place_name(anchor) for anchor in anchors]
    if any(anchor == normalized_attraction or anchor in normalized_attraction for anchor in normalized_anchors):
        return True
    if "wutai" in normalized_anchors:
        return "wutai" in normalized_attraction or "taihuai" in normalized_attraction
    return True


def _validate_plan(request: TripRequest, plan: TripPlan, mcp_results: McpResults) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    route_by_key = {_route_result_key(route): route for route in mcp_results.routes}
    planned_places = [
        item.stay.place_name
        for day in plan.days
        for item in day.timeline
        if item.stay and item.stay.purpose == StayPurpose.VISIT
    ]

    _validate_route_segments_have_mcp_data(plan, route_by_key, issues)

    for required_place in request.must_visit:
        if not _place_matches_any(required_place, planned_places):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.MISSING_MUST_VISIT,
                    severity=Severity.HIGH,
                    locations=[required_place],
                    reason=f"Must-visit place {required_place} is not present in the timeline.",
                    suggested_action="Add the must-visit as a stay item or explain infeasibility.",
                )
            )

    for day in plan.days:
        _validate_timeline_order(day, issues)
        _validate_timeline_continuity(day, issues)
        if day.accommodation_area and _is_unverified_lodging_label(day.accommodation_area):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.MISSING_MCP_DATA,
                    severity=Severity.HIGH,
                    date=day.date,
                    locations=[day.accommodation_area],
                    reason="No verified lodging candidate with usable location data is available for this day.",
                    suggested_action="Fetch a real lodging candidate near the main anchor before finalizing the plan.",
                )
            )
        covered_minutes = _covered_timeline_minutes(day.timeline)
        if covered_minutes < 20 * 60:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.INCOMPLETE_DAY_TIMELINE,
                    severity=Severity.MEDIUM,
                    day=day.day,
                    date=day.date,
                    locations=[day.city],
                    reason=f"Timeline covers only {covered_minutes} minutes of the day.",
                    suggested_action="Add sleep, meals, moves, rest, and buffers until the day is approximately covered.",
                )
            )
        for item in day.timeline:
            if item.stay and item.stay.purpose == StayPurpose.VISIT:
                _validate_visit_item(day, item, request, mcp_results, issues)
            if item.move:
                _validate_move_item(day, item, request, route_by_key, issues)

    if request.origin != request.destination:
        _validate_trip_boundary_and_flow(request, plan, issues)
        has_return = any(
            item.move and item.move.purpose == MovePurpose.RETURN and _place_matches_any(request.origin, [item.move.destination])
            for day in plan.days
            for item in day.timeline
        )
        if not has_return:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.MISSING_RETURN_TRANSFER,
                    severity=Severity.HIGH,
                    day=plan.days[-1].day if plan.days else None,
                    date=plan.days[-1].date if plan.days else None,
                    locations=[request.origin],
                    reason="The timeline has no explicit return move back to the origin.",
                    suggested_action="Add a final-day return move with departure and arrival time.",
                )
            )
        _validate_return_position(request, plan, issues)

    return issues


def _validate_route_segments_have_mcp_data(
    plan: TripPlan,
    route_by_key: dict[tuple[str, str, TransportMode, str, str], RouteResult],
    issues: list[ValidationIssue],
) -> None:
    for segment in plan.route_segments:
        route = _find_route_result(
            route_by_key,
            segment.origin,
            segment.destination,
            segment.mode,
            segment.origin_city,
            segment.destination_city,
        )
        if route is not None:
            continue
        issues.append(
            ValidationIssue(
                issue_type=IssueType.MISSING_MCP_DATA,
                severity=Severity.HIGH,
                day=None,
                date=segment.departure_date,
                locations=[segment.origin, segment.destination],
                reason=(
                    "Route segment has no matching MCP route result; "
                    f"{segment.origin} -> {segment.destination} by {segment.mode.value} is unverified."
                ),
                suggested_action="Re-query this route with concrete origin_city and destination_city, or mark the plan as not executable.",
            )
        )


def _validate_visit_item(
    day: PlanDay,
    item: TimelineItem,
    request: TripRequest,
    mcp_results: McpResults,
    issues: list[ValidationIssue],
) -> None:
    assert item.stay is not None
    if item.stay.category == PlaceCategory.OUTDOOR and _is_bad_weather(mcp_results, day.city, day.date):
        is_must_visit = any(_place_matches_any(required, [item.stay.place_name]) for required in request.must_visit)
        issues.append(
            ValidationIssue(
                issue_type=IssueType.BAD_WEATHER,
                severity=Severity.MEDIUM if is_must_visit else Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.stay.place_name],
                reason=f"{item.stay.place_name} is outdoor while {day.city} has bad weather on {day.date}.",
                suggested_action=(
                    "Keep only if the user accepts weather risk, or move it to a better-weather day."
                    if is_must_visit
                    else "Move the outdoor visit to a better-weather day or replace it with an indoor stay."
                ),
            )
        )

    detail = _attraction_detail_for(mcp_results, item.stay.place_name, day.city, day.date)
    if detail and not detail.is_open:
        issues.append(
            ValidationIssue(
                issue_type=IssueType.ATTRACTION_CLOSED,
                severity=Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.stay.place_name],
                reason=f"{item.stay.place_name} is closed on {day.date}.",
                suggested_action="Move the visit to another date or choose an open replacement.",
            )
        )


def _validate_move_item(
    day: PlanDay,
    item: TimelineItem,
    request: TripRequest,
    route_by_key: dict[tuple[str, str, TransportMode, str, str], RouteResult],
    issues: list[ValidationIssue],
) -> None:
    assert item.move is not None
    route = _find_route_result(
        route_by_key,
        item.move.origin,
        item.move.destination,
        item.move.mode,
        item.move.origin_city,
        item.move.destination_city,
    )
    if route is None:
        issues.append(
            ValidationIssue(
                issue_type=IssueType.MISSING_MCP_DATA,
                severity=Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason=(
                    "Timeline move has no matching MCP route result; "
                    f"{item.move.origin} -> {item.move.destination} by {item.move.mode.value} is unverified."
                ),
                suggested_action="Do not use guessed durations; collect MCP route data or mark this transfer as unverified.",
            )
        )
    duration = item.move.duration_minutes or item.duration_minutes
    distance = item.move.distance_km
    cross_city = _move_crosses_cities(item.move)
    explicit_station_or_lodging_transfer = _is_station_or_lodging_transfer(item.move, day)
    if cross_city and 0 < distance < 50:
        issues.append(
            ValidationIssue(
                issue_type=IssueType.INFEASIBLE_PLAN,
                severity=Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason=(
                    f"Cross-city route result is suspicious: "
                    f"{item.move.origin_city or item.move.origin} -> "
                    f"{item.move.destination_city or item.move.destination} is only {distance} km."
                ),
                suggested_action="Re-query route with explicit origin_city and destination_city, then rebuild this transfer.",
            )
        )
    if not cross_city and item.move.purpose == MovePurpose.LOCAL and distance > 120 and not explicit_station_or_lodging_transfer:
        issues.append(
            ValidationIssue(
                issue_type=IssueType.ROUTE_TOO_LONG,
                severity=Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason=f"Local route distance is suspiciously long at {distance} km.",
                suggested_action="Choose lodging and visits within the same city area, or convert this to an intercity transfer.",
            )
        )
    if item.move.purpose == MovePurpose.LOCAL and duration > 90 and not explicit_station_or_lodging_transfer:
        severity = Severity.HIGH if duration > 120 else Severity.MEDIUM
        issues.append(
            ValidationIssue(
                issue_type=IssueType.ROUTE_TOO_LONG,
                severity=severity,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason=f"Local move takes {duration} minutes.",
                suggested_action="Choose closer lodging, reorder nearby stays, or split this part into another city/day.",
            )
        )
    if item.move.purpose in {MovePurpose.OUTBOUND, MovePurpose.INTERCITY, MovePurpose.RETURN} and duration > 360:
        severity = Severity.HIGH if duration > 720 else Severity.MEDIUM
        issues.append(
            ValidationIssue(
                issue_type=IssueType.ROUTE_TOO_LONG,
                severity=severity,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason=f"Intercity move takes {duration} minutes.",
                suggested_action="Reserve more of the day for transfer or choose a nearer base city.",
            )
        )
    if item.move.purpose == MovePurpose.LOCAL and day.accommodation_area and not explicit_station_or_lodging_transfer:
        touches_lodging = _locations_equivalent(day.accommodation_area, item.move.origin) or _locations_equivalent(
            day.accommodation_area,
            item.move.destination,
        )
        if touches_lodging and duration > 60:
            severity = Severity.HIGH if duration > 120 else Severity.MEDIUM
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.LODGING_TOO_FAR,
                    severity=severity,
                    day=day.day,
                    date=day.date,
                    locations=[day.accommodation_area, item.move.origin, item.move.destination],
                    reason=f"Lodging-related move takes {duration} minutes.",
                    suggested_action="Choose lodging near the main anchor for this day.",
                )
            )
    if _is_vague_route_endpoint(item.move.origin) or _is_vague_route_endpoint(item.move.destination):
        issues.append(
            ValidationIssue(
                issue_type=IssueType.ROUTE_ENDPOINT_TOO_VAGUE,
                severity=Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason="Move endpoint is too broad for route validation.",
                suggested_action="Use concrete stations, lodging names, or attraction names as endpoints.",
            )
        )
    if (
        request.travelers.has_children_or_infants
        and item.move.purpose == MovePurpose.LOCAL
        and duration > 90
        and not explicit_station_or_lodging_transfer
    ):
        issues.append(
            ValidationIssue(
                issue_type=IssueType.LONG_TRANSFER_WITH_CHILDREN,
                severity=Severity.HIGH,
                day=day.day,
                date=day.date,
                locations=[item.move.origin, item.move.destination],
                reason=f"Local transfer with children takes {duration} minutes.",
                suggested_action="Reduce local transfer duration or add a nearby overnight stay.",
            )
        )


def _validate_timeline_order(day: PlanDay, issues: list[ValidationIssue]) -> None:
    ordered = sorted(day.timeline, key=lambda item: item.sequence)
    previous: TimelineItem | None = None
    for item in ordered:
        if previous and item.start_time < previous.end_time:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.TIME_CONFLICT,
                    severity=Severity.CRITICAL,
                    day=day.day,
                    date=day.date,
                    locations=[day.city],
                    reason=f"Timeline item {item.sequence} overlaps with item {previous.sequence}.",
                    suggested_action="Recalculate start/end times so timeline items are sequential.",
                )
            )
        previous = item


def _validate_timeline_continuity(day: PlanDay, issues: list[ValidationIssue]) -> None:
    ordered = sorted(day.timeline, key=lambda item: item.sequence)
    for previous, current in zip(ordered, ordered[1:]):
        if (
            previous.move
            and current.stay
            and _should_validate_stay_location(current.stay)
            and not _place_matches_any(previous.move.destination, _stay_location_names(current.stay, day))
        ):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.INFEASIBLE_PLAN,
                    severity=Severity.HIGH,
                    day=day.day,
                    date=day.date,
                    locations=[previous.move.destination, current.stay.place_name],
                    reason="Move destination does not match the following stay location.",
                    suggested_action="Rebuild the timeline so every move arrives at the next stay location.",
                )
            )
        if (
            previous.stay
            and current.move
            and _should_validate_stay_location(previous.stay)
            and not _place_matches_any(current.move.origin, _stay_location_names(previous.stay, day))
        ):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.INFEASIBLE_PLAN,
                    severity=Severity.HIGH,
                    day=day.day,
                    date=day.date,
                    locations=[previous.stay.place_name, current.move.origin],
                    reason="Move origin does not match the previous stay location.",
                    suggested_action="Rebuild the timeline so every move starts from the previous stay location.",
                )
            )
        if previous.move and current.move and not _location_names_match(
            _timeline_primary_end_location_names(previous, day),
            _timeline_primary_start_location_names(current, day),
        ):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.INFEASIBLE_PLAN,
                    severity=Severity.HIGH,
                    day=day.day,
                    date=day.date,
                    locations=[previous.move.destination, current.move.origin],
                    reason="Adjacent moves do not connect: previous destination differs from next origin.",
                    suggested_action="Insert a local transfer or correct move endpoints before using this timeline.",
                )
            )
        if (
            previous.stay
            and current.stay
            and _should_validate_stay_location(previous.stay)
            and _should_validate_stay_location(current.stay)
            and not _location_names_match(
                _timeline_primary_end_location_names(previous, day),
                _timeline_primary_start_location_names(current, day),
            )
        ):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.INFEASIBLE_PLAN,
                    severity=Severity.HIGH,
                    day=day.day,
                    date=day.date,
                    locations=[previous.stay.place_name, current.stay.place_name],
                    reason="Adjacent stays do not connect and no move is scheduled between them.",
                    suggested_action="Insert a move item or merge stays at the same concrete location.",
                )
            )


def _validate_trip_boundary_and_flow(request: TripRequest, plan: TripPlan, issues: list[ValidationIssue]) -> None:
    items = _chronological_timeline_items(plan)
    if not items:
        return

    first_day, first_item = items[0]
    if not _location_names_match([request.origin], _timeline_start_location_names(first_item, first_day)):
        issues.append(
            ValidationIssue(
                issue_type=IssueType.INFEASIBLE_PLAN,
                severity=Severity.CRITICAL,
                day=first_day.day,
                date=first_day.date,
                locations=[request.origin, *_timeline_start_location_names(first_item, first_day)],
                reason="The first chronological timeline item does not start at the trip origin.",
                suggested_action="Start day 1 at the user's origin, then add the outbound move before destination lodging or visits.",
            )
        )

    last_day, last_item = items[-1]
    if not _location_names_match([request.origin], _timeline_end_location_names(last_item, last_day)):
        issues.append(
            ValidationIssue(
                issue_type=IssueType.MISSING_RETURN_TRANSFER,
                severity=Severity.CRITICAL,
                day=last_day.day,
                date=last_day.date,
                locations=[request.origin, *_timeline_end_location_names(last_item, last_day)],
                reason="The last chronological timeline item does not end at the trip origin.",
                suggested_action="End the itinerary at the user's origin after an explicit return move.",
            )
        )

    for (previous_day, previous_item), (current_day, current_item) in zip(items, items[1:]):
        previous_locations = _timeline_primary_end_location_names(previous_item, previous_day)
        current_locations = _timeline_primary_start_location_names(current_item, current_day)
        if _location_names_match(previous_locations, current_locations):
            continue
        issues.append(
            ValidationIssue(
                issue_type=IssueType.INFEASIBLE_PLAN,
                severity=Severity.HIGH,
                day=current_day.day,
                date=current_day.date,
                locations=[*previous_locations, *current_locations],
                reason="Adjacent timeline items do not connect: previous end location differs from next start location.",
                suggested_action="Insert a move item or correct endpoints so each item starts where the previous item ended.",
            )
        )


def _validate_return_position(request: TripRequest, plan: TripPlan, issues: list[ValidationIssue]) -> None:
    returned = False
    for day in sorted(plan.days, key=lambda current: (current.date, current.day)):
        for item in sorted(day.timeline, key=lambda current: current.sequence):
            if returned and not _timeline_item_is_at_origin(item, request.origin):
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.MISSING_RETURN_TRANSFER,
                        severity=Severity.HIGH,
                        day=day.day,
                        date=day.date,
                        locations=[request.origin, day.city],
                        reason="Timeline continues away from the origin after the return move.",
                        suggested_action="Place the return move at the true end of the trip or remove later destination activities.",
                    )
                )
                return
            if item.move and item.move.purpose == MovePurpose.RETURN and _place_matches_any(request.origin, [item.move.destination]):
                returned = True


def _timeline_item_is_at_origin(item: TimelineItem, origin: str) -> bool:
    if item.move:
        return _place_matches_any(origin, [item.move.origin, item.move.origin_city])
    if item.stay:
        return _place_matches_any(origin, [item.stay.place_name, item.stay.city])
    return False


def _chronological_timeline_items(plan: TripPlan) -> list[tuple[PlanDay, TimelineItem]]:
    return _chronological_day_items(plan.days)


def _chronological_day_items(days: list[PlanDay]) -> list[tuple[PlanDay, TimelineItem]]:
    pairs: list[tuple[PlanDay, TimelineItem]] = []
    for day in sorted(days, key=lambda current: (current.date, current.day)):
        for item in sorted(day.timeline, key=lambda current: (_time_to_minutes(current.start_time), current.sequence)):
            pairs.append((day, item))
    return pairs


def _timeline_start_location_names(item: TimelineItem, day: PlanDay) -> list[str]:
    if item.move:
        return [name for name in [item.move.origin, item.move.origin_city] if name]
    if item.stay:
        return _stay_location_names(item.stay, day)
    return [day.city]


def _timeline_primary_start_location_names(item: TimelineItem, day: PlanDay) -> list[str]:
    if item.move:
        return [item.move.origin]
    if item.stay:
        return [item.stay.place_name]
    return [day.city]


def _timeline_end_location_names(item: TimelineItem, day: PlanDay) -> list[str]:
    if item.move:
        return [name for name in [item.move.destination, item.move.destination_city] if name]
    if item.stay:
        return _stay_location_names(item.stay, day)
    return [day.city]


def _timeline_primary_end_location_names(item: TimelineItem, day: PlanDay) -> list[str]:
    if item.move:
        return [item.move.destination]
    if item.stay:
        return [item.stay.place_name]
    return [day.city]


def _location_names_match(first: list[str], second: list[str]) -> bool:
    return any(_locations_equivalent(left, right) for left in first for right in second if left and right)


def _locations_equivalent(left: str, right: str) -> bool:
    normalized_left = _normalize_place_name(left)
    normalized_right = _normalize_place_name(right)
    return bool(normalized_left and normalized_right and normalized_left == normalized_right)


def _stay_location_names(stay: StayDetail, day: PlanDay) -> list[str]:
    names = [stay.place_name, stay.city]
    if stay.purpose in {StayPurpose.SLEEP, StayPurpose.REST, StayPurpose.HOTEL_CHECKIN, StayPurpose.HOTEL_CHECKOUT}:
        names.extend([day.accommodation_area or "", day.overnight_accommodation or ""])
    return [name for name in names if name]


def _should_validate_stay_location(stay: StayDetail) -> bool:
    return stay.purpose in {
        StayPurpose.VISIT,
        StayPurpose.SLEEP,
        StayPurpose.HOTEL_CHECKIN,
        StayPurpose.HOTEL_CHECKOUT,
    }


def _move_crosses_cities(move: MoveDetail) -> bool:
    if move.purpose in {MovePurpose.OUTBOUND, MovePurpose.INTERCITY, MovePurpose.RETURN}:
        return True
    return bool(move.origin_city and move.destination_city and move.origin_city != move.destination_city)


def _apply_route_results_to_plan(
    plan: TripPlan,
    route_by_key: dict[tuple[str, str, TransportMode, str, str], RouteResult],
) -> None:
    for segment in plan.route_segments:
        route = _find_route_result(
            route_by_key,
            segment.origin,
            segment.destination,
            segment.mode,
            segment.origin_city,
            segment.destination_city,
        )
        if route:
            segment.estimated_duration_minutes = route.duration_minutes
            segment.estimated_distance_km = route.distance_km
            segment.notes = _append_note(segment.notes, "Updated from MCP route data.")

    for day in plan.days:
        for item in day.timeline:
            if not item.move:
                continue
            route = _find_route_result(
                route_by_key,
                item.move.origin,
                item.move.destination,
                item.move.mode,
                item.move.origin_city,
                item.move.destination_city,
            )
            if route is None:
                continue
            item.move.duration_minutes = route.duration_minutes
            item.move.distance_km = route.distance_km
            item.move.notes = _append_note(item.move.notes, "Updated from MCP route data.")
            item.end_time = _shift_time(item.start_time, route.duration_minutes)
        _repair_day_timeline_continuity(day)
        _remove_unverified_lodging_moves(day)
        _remove_same_location_moves(day)
        _fit_day_timeline_sequentially(day)
        _recalculate_day_totals(day)
    _recalculate_plan_totals(plan)


def _apply_route_results_to_segments(segments: list[TripSegment], mcp_results: McpResults) -> None:
    route_by_key = {_route_result_key(route): route for route in mcp_results.routes}
    for segment in segments:
        route = _find_route_result(
            route_by_key,
            segment.origin,
            segment.destination,
            segment.mode,
            segment.origin_city,
            segment.destination_city,
        )
        if route:
            segment.estimated_duration_minutes = route.duration_minutes
            segment.estimated_distance_km = route.distance_km
            if segment.departure_time:
                segment.arrival_time = _shift_time(segment.departure_time, route.duration_minutes)


def _recalculate_day_totals(day: PlanDay) -> None:
    day.total_stay_minutes = 0
    day.total_move_minutes = 0
    day.total_sleep_minutes = 0
    day.estimated_cost = 0
    for item in day.timeline:
        if item.stay:
            item.stay.duration_minutes = item.duration_minutes
            day.total_stay_minutes += item.duration_minutes
            if item.stay.purpose == StayPurpose.SLEEP:
                day.total_sleep_minutes += item.duration_minutes
            day.estimated_cost += item.stay.estimated_cost
        if item.move:
            item.move.duration_minutes = item.duration_minutes if not item.move.duration_minutes else item.move.duration_minutes
            day.total_move_minutes += item.move.duration_minutes
            day.estimated_cost += item.move.estimated_cost


def _fit_day_timeline_sequentially(day: PlanDay) -> None:
    fitted: list[TimelineItem] = []
    cursor = 0
    for item in sorted(day.timeline, key=lambda current: (_time_to_minutes(current.start_time), current.sequence)):
        original_duration = max(1, item.move.duration_minutes if item.move else item.duration_minutes)
        start = max(_time_to_minutes(item.start_time), cursor)
        end = start + original_duration
        if end > 23 * 60 + 59:
            if item.stay and item.stay.purpose in {StayPurpose.REST, StayPurpose.BUFFER, StayPurpose.MEAL}:
                continue
            end = 23 * 60 + 59
        if end <= start:
            continue
        item.start_time = _minutes_to_time(start)
        item.end_time = _minutes_to_time(end)
        if item.move:
            item.move.duration_minutes = end - start
        if item.stay:
            item.stay.duration_minutes = end - start
        fitted.append(item)
        cursor = end
    day.timeline = fitted
    _resequence_timeline(day.timeline)


def _repair_day_timeline_continuity(day: PlanDay) -> None:
    ordered = sorted(day.timeline, key=lambda current: (_time_to_minutes(current.start_time), current.sequence))
    repaired: list[TimelineItem] = []
    for item in ordered:
        if not repaired:
            repaired.append(item)
            continue

        previous = repaired[-1]
        previous_locations = _timeline_primary_end_location_names(previous, day)
        current_locations = _timeline_primary_start_location_names(item, day)
        if _location_names_match(previous_locations, current_locations):
            repaired.append(item)
            continue

        previous_location = _first_location(previous_locations, day.city)
        current_location = _first_location(current_locations, day.city)
        if not previous_location or not current_location or _place_matches_any(previous_location, [current_location]):
            repaired.append(item)
            continue
        if _is_unverified_lodging_label(previous_location) or _is_unverified_lodging_label(current_location):
            repaired.append(item)
            continue

        if item.move and item.move.purpose == MovePurpose.LOCAL and not (previous.move and _move_crosses_cities(previous.move)):
            item.move.origin = previous_location
            item.move.origin_city = _timeline_primary_end_city(previous, day)
            repaired.append(item)
            continue

        if previous.move and previous.move.purpose == MovePurpose.LOCAL:
            previous.move.destination = current_location
            previous.move.destination_city = _timeline_primary_start_city(item, day)
            repaired.append(item)
            continue

        if item.stay and _is_passive_stay(item.stay):
            _move_stay_to_location(item.stay, previous_location, "Adjusted to preserve timeline location continuity.")
            repaired.append(item)
            continue

        connector = _local_connector_after_previous(previous, item, day, previous_location, current_location)
        if connector is not None:
            repaired.append(connector)
        repaired.append(item)

    day.timeline = repaired
    _resequence_timeline(day.timeline)


def _repair_cross_day_timeline_continuity(plan: TripPlan) -> None:
    days = sorted(plan.days, key=lambda current: (current.date, current.day))
    for previous_day, current_day in zip(days, days[1:]):
        previous_items = sorted(previous_day.timeline, key=lambda current: (_time_to_minutes(current.start_time), current.sequence))
        current_items = sorted(current_day.timeline, key=lambda current: (_time_to_minutes(current.start_time), current.sequence))
        if not previous_items or not current_items:
            continue

        previous = previous_items[-1]
        current = current_items[0]
        previous_locations = _timeline_primary_end_location_names(previous, previous_day)
        current_locations = _timeline_primary_start_location_names(current, current_day)
        if _location_names_match(previous_locations, current_locations):
            continue

        previous_location = _first_location(previous_locations, previous_day.city)
        current_location = _first_location(current_locations, current_day.city)
        if current.stay and _is_passive_stay(current.stay):
            _move_stay_to_location(current.stay, previous_location, "Adjusted overnight location from previous day.")
            continue
        if current.move and current.move.purpose == MovePurpose.LOCAL:
            current.move.origin = previous_location
            current.move.origin_city = _timeline_primary_end_city(previous, previous_day)
            continue
        connector = _local_connector_between_locations(
            sequence=1,
            start_minute=0,
            origin=previous_location,
            destination=current_location,
            origin_city=_timeline_primary_end_city(previous, previous_day),
            destination_city=_timeline_primary_start_city(current, current_day),
            city=current_day.city,
            notes="Inserted deterministic cross-day continuity transfer.",
        )
        if connector is not None:
            current_day.timeline.insert(0, connector)
            _resequence_timeline(current_day.timeline)


def _remove_same_location_moves(day: PlanDay) -> None:
    day.timeline = [
        item
        for item in day.timeline
        if not (
            item.move
            and item.move.purpose == MovePurpose.LOCAL
            and _locations_equivalent(item.move.origin, item.move.destination)
        )
    ]
    _resequence_timeline(day.timeline)


def _remove_unverified_lodging_moves(day: PlanDay) -> None:
    day.timeline = [
        item
        for item in day.timeline
        if not (item.move and item.move.purpose == MovePurpose.LOCAL and _move_has_unverified_endpoint(item.move))
    ]
    _resequence_timeline(day.timeline)


def _local_connector_after_previous(
    previous: TimelineItem,
    current: TimelineItem,
    day: PlanDay,
    origin: str,
    destination: str,
) -> TimelineItem | None:
    start_minute = _time_to_minutes(previous.end_time)
    if start_minute >= 23 * 60 + 59:
        return None
    return _local_connector_between_locations(
        sequence=previous.sequence + 1,
        start_minute=start_minute,
        origin=origin,
        destination=destination,
        origin_city=_timeline_primary_end_city(previous, day),
        destination_city=_timeline_primary_start_city(current, day),
        city=day.city,
        notes="Inserted deterministic continuity transfer.",
    )


def _local_connector_between_locations(
    sequence: int,
    start_minute: int,
    origin: str,
    destination: str,
    origin_city: str,
    destination_city: str,
    city: str,
    notes: str,
) -> TimelineItem | None:
    if _place_matches_any(origin, [destination]):
        return None
    duration = min(25, max(1, 23 * 60 + 59 - start_minute))
    return TimelineItem(
        sequence=sequence,
        item_type=TimelineItemType.MOVE,
        start_time=_minutes_to_time(start_minute),
        end_time=_minutes_to_time(start_minute + duration),
        city=city,
        move=MoveDetail(
            origin=origin,
            destination=destination,
            origin_city=origin_city,
            destination_city=destination_city,
            mode=TransportMode.TAXI,
            purpose=MovePurpose.LOCAL,
            duration_minutes=duration,
            estimated_cost=30,
            notes=notes,
        ),
    )


def _first_location(names: list[str], fallback: str) -> str:
    return next((name for name in names if name), fallback)


def _timeline_primary_end_city(item: TimelineItem, day: PlanDay) -> str:
    if item.move:
        return item.move.destination_city or item.city or day.city
    if item.stay:
        return item.stay.city or item.city or day.city
    return item.city or day.city


def _timeline_primary_start_city(item: TimelineItem, day: PlanDay) -> str:
    if item.move:
        return item.move.origin_city or item.city or day.city
    if item.stay:
        return item.stay.city or item.city or day.city
    return item.city or day.city


def _sync_local_move_endpoints(day: PlanDay) -> None:
    ordered = sorted(day.timeline, key=lambda item: item.sequence)
    for index, item in enumerate(ordered):
        if not item.move or item.move.purpose != MovePurpose.LOCAL:
            continue
        if day.accommodation_area and _locations_equivalent(item.move.destination, day.accommodation_area):
            continue
        if _is_explicit_lodging_or_station_transfer(item.move):
            continue
        previous_stay = next((candidate.stay for candidate in reversed(ordered[:index]) if candidate.stay), None)
        next_stay = next((candidate.stay for candidate in ordered[index + 1 :] if candidate.stay), None)
        if previous_stay and previous_stay.purpose == StayPurpose.VISIT:
            item.move.origin = previous_stay.place_name
        elif day.accommodation_area and index <= 4:
            item.move.origin = day.accommodation_area
        if next_stay and next_stay.purpose == StayPurpose.VISIT:
            item.move.destination = next_stay.place_name
        elif day.accommodation_area and previous_stay and previous_stay.purpose == StayPurpose.VISIT:
            item.move.destination = day.accommodation_area


def _is_explicit_lodging_or_station_transfer(move: MoveDetail) -> bool:
    notes = move.notes.lower()
    return "arrival point to lodging" in notes or "lodging to return departure point" in notes


def _is_station_or_lodging_transfer(move: MoveDetail, day: PlanDay) -> bool:
    if _is_explicit_lodging_or_station_transfer(move):
        return True
    if move.purpose != MovePurpose.LOCAL or not day.accommodation_area:
        return False
    touches_lodging = _locations_equivalent(day.accommodation_area, move.origin) or _locations_equivalent(
        day.accommodation_area,
        move.destination,
    )
    touches_day_anchor = _locations_equivalent(day.city, move.origin) or _locations_equivalent(day.city, move.destination)
    return touches_lodging and touches_day_anchor


def _endpoint_is_lodging_reference(endpoint: str, accommodation_area: str | None) -> bool:
    normalized = _normalize_place_name(endpoint)
    if normalized in {"hotel", "lodging", "accommodation"}:
        return True
    return bool(accommodation_area and _locations_equivalent(endpoint, accommodation_area))


def _normalize_passive_boundary_stays(day: PlanDay, request: TripRequest, days: list[PlanDay]) -> None:
    items = _chronological_day_items(days)
    outbound_seen = False
    return_seen = False
    for current_day, item in items:
        if item.move and item.move.purpose == MovePurpose.OUTBOUND:
            outbound_seen = True
        if item.move and item.move.purpose == MovePurpose.RETURN and _place_matches_any(request.origin, [item.move.destination]):
            return_seen = True
            continue
        if current_day is not day or not item.stay or not _is_passive_stay(item.stay):
            continue
        if not outbound_seen:
            _move_stay_to_location(item.stay, request.origin, "Pre-trip rest at origin before outbound transfer.")
        elif return_seen:
            _move_stay_to_location(item.stay, request.origin, "Post-return rest at origin.")


def _normalize_passive_stay_locations(day: PlanDay) -> None:
    current_location = ""
    current_city = day.city
    for item in sorted(day.timeline, key=lambda current: (_time_to_minutes(current.start_time), current.sequence)):
        if item.move:
            if (
                item.move.purpose == MovePurpose.LOCAL
                and current_location
                and not _place_matches_any(item.move.origin, [current_location, current_city])
            ):
                item.move.origin = current_location
                item.move.origin_city = current_city
            current_location = item.move.destination
            current_city = item.move.destination_city or item.city
            continue
        if not item.stay:
            continue
        if current_location and _is_passive_stay(item.stay) and not _place_matches_any(
            item.stay.place_name,
            [current_location, current_city],
        ):
            item.stay.place_name = current_location
            item.stay.city = current_city
        current_location = item.stay.place_name
        current_city = item.stay.city or item.city


def _is_passive_stay(stay: StayDetail) -> bool:
    return stay.purpose in {
        StayPurpose.SLEEP,
        StayPurpose.MEAL,
        StayPurpose.REST,
        StayPurpose.BUFFER,
        StayPurpose.HOTEL_CHECKOUT,
    }


def _move_stay_to_location(stay: StayDetail, location: str, note: str) -> None:
    stay.place_name = location
    stay.city = location
    stay.notes = f"{stay.notes} {note}".strip()


def _fill_segment_city_context(segment: TripSegment, city_route_plan: CityRoutePlan, request: TripRequest) -> None:
    if segment.segment_type == SegmentType.OUTBOUND:
        segment.origin_city = segment.origin_city or request.origin
        segment.destination_city = segment.destination_city or (
            city_route_plan.stays[0].city if city_route_plan.stays else segment.destination
        )
    elif segment.segment_type == SegmentType.RETURN:
        segment.origin_city = segment.origin_city or (
            city_route_plan.stays[-1].city if city_route_plan.stays else segment.origin
        )
        segment.destination_city = segment.destination_city or request.origin
    else:
        segment.origin_city = segment.origin_city or segment.origin
        segment.destination_city = segment.destination_city or segment.destination
    segment.origin_city = _route_lookup_city(segment.origin_city, segment.origin, request)
    segment.destination_city = _route_lookup_city(segment.destination_city, segment.destination, request)


def _fill_day_move_city_context(day: PlanDay, request: TripRequest, city_route_plan: CityRoutePlan) -> None:
    day_lookup_city = _route_lookup_city(day.city, day.city, request)
    for item in day.timeline:
        if not item.move:
            continue
        move = item.move
        if move.purpose == MovePurpose.LOCAL:
            move.origin_city = _usable_city_context(move.origin_city, move.origin, day_lookup_city)
            move.destination_city = _usable_city_context(move.destination_city, move.destination, day_lookup_city)
        elif move.purpose == MovePurpose.OUTBOUND:
            move.origin_city = move.origin_city or request.origin
            move.destination_city = move.destination_city or day_lookup_city
        elif move.purpose == MovePurpose.RETURN:
            move.origin_city = move.origin_city or day_lookup_city
            move.destination_city = move.destination_city or request.origin
        elif move.purpose == MovePurpose.INTERCITY:
            move.origin_city = move.origin_city or move.origin
            move.destination_city = move.destination_city or move.destination
        move.origin_city = _route_lookup_city(move.origin_city, move.origin, request)
        move.destination_city = _route_lookup_city(move.destination_city, move.destination, request)


def _usable_city_context(current_city: str, endpoint: str, fallback_city: str) -> str:
    if not current_city or _place_matches_any(current_city, [endpoint]):
        return fallback_city
    return current_city


def _route_lookup_city(city: str, endpoint: str, request: TripRequest) -> str:
    if _place_matches_any(request.origin, [endpoint, city]):
        return request.origin
    mapped = _anchor_city_for_place(request.destination, endpoint)
    if mapped and mapped != request.destination:
        return mapped
    if city and not _is_vague_route_endpoint(city):
        return city
    return endpoint or city


def _recalculate_plan_totals(plan: TripPlan) -> None:
    for day in plan.days:
        _recalculate_day_totals(day)
    plan.total_estimated_cost = sum(day.estimated_cost for day in plan.days)
    for stay in plan.accommodations:
        nights = max(1, (stay.check_out_date - stay.check_in_date).days)
        plan.total_estimated_cost += stay.estimated_cost_per_night * nights


def _route_query_from_segment(segment: TripSegment, stage: McpQueryStage) -> McpQuery:
    return McpQuery(
        tool_name=McpToolName.GET_ROUTE_TIME,
        args={
            "origin": segment.origin,
            "destination": segment.destination,
            "origin_city": segment.origin_city or segment.origin,
            "destination_city": segment.destination_city or segment.destination,
            "mode": segment.mode.value,
        },
        purpose=f"Verify {segment.segment_type.value} segment duration.",
        stage=stage,
    )


def _route_query_from_move(move: MoveDetail, stage: McpQueryStage) -> McpQuery:
    return McpQuery(
        tool_name=McpToolName.GET_ROUTE_TIME,
        args={
            "origin": move.origin,
            "destination": move.destination,
            "origin_city": move.origin_city or move.origin,
            "destination_city": move.destination_city or move.destination,
            "mode": move.mode.value,
        },
        purpose=f"Verify {move.purpose.value} move duration.",
        stage=stage,
    )


def _accommodation_locations_by_name(plan: TripPlan) -> dict[str, str]:
    locations: dict[str, str] = {}
    for accommodation in plan.accommodations:
        if accommodation.hotel_name and accommodation.location:
            locations[accommodation.hotel_name] = accommodation.location
        if accommodation.area and accommodation.location:
            locations[accommodation.area] = accommodation.location
    return locations


def _attach_known_endpoint_locations(query: McpQuery, endpoint_locations: dict[str, str]) -> None:
    origin = str(query.args.get("origin", ""))
    destination = str(query.args.get("destination", ""))
    if origin in endpoint_locations:
        query.args["origin_location"] = endpoint_locations[origin]
    if destination in endpoint_locations:
        query.args["destination_location"] = endpoint_locations[destination]


def _route_result_key(route: RouteResult) -> tuple[str, str, TransportMode, str, str]:
    return (
        route.origin,
        route.destination,
        route.mode,
        route.origin_city,
        route.destination_city,
    )


def _find_route_result(
    route_by_key: dict[tuple[str, str, TransportMode, str, str], RouteResult],
    origin: str,
    destination: str,
    mode: TransportMode,
    origin_city: str,
    destination_city: str,
) -> RouteResult | None:
    return route_by_key.get((origin, destination, mode, origin_city, destination_city)) or route_by_key.get(
        (origin, destination, mode, "", "")
    )


def _route_matches_query(route: RouteResult, args: dict[str, Any]) -> bool:
    mode = TransportMode(str(args.get("mode", TransportMode.TAXI.value)))
    if route.origin != str(args["origin"]) or route.destination != str(args["destination"]) or route.mode != mode:
        return False
    return _city_context_matches(route.origin_city, str(args.get("origin_city", ""))) and _city_context_matches(
        route.destination_city,
        str(args.get("destination_city", "")),
    )


def _city_context_matches(result_city: str, query_city: str) -> bool:
    return not result_city or not query_city or result_city == query_city


def _dedupe_mcp_queries(queries: list[McpQuery]) -> list[McpQuery]:
    seen: set[tuple[str, str]] = set()
    deduped: list[McpQuery] = []
    for query in queries:
        key = (query.tool_name.value, _stable_args_key(query.args))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def _filter_queries_with_existing_results(queries: list[McpQuery], mcp_results: McpResults) -> list[McpQuery]:
    return [query for query in queries if not _query_result_exists(query, mcp_results)]


def _query_result_exists(query: McpQuery, mcp_results: McpResults) -> bool:
    args = query.args
    if query.tool_name == McpToolName.GET_WEATHER:
        query_date = _parse_date(str(args["date"]))
        return any(item.city == str(args["city"]) and item.date == query_date for item in mcp_results.weather)
    if query.tool_name == McpToolName.GET_ROUTE_TIME:
        if args.get("origin_location") or args.get("destination_location"):
            return False
        return any(_route_matches_query(item, args) for item in mcp_results.routes)
    if query.tool_name == McpToolName.GET_ATTRACTION_DETAIL:
        query_date = _parse_date(str(args["date"]))
        query_city = str(args.get("city", ""))
        return any(
            item.date == query_date
            and (not query_city or item.city == query_city)
            and _place_matches_any(str(args["name"]), [item.name])
            for item in mcp_results.attractions
        )
    if query.tool_name == McpToolName.SEARCH_ATTRACTIONS:
        return any(item.city == str(args["city"]) and item.date is None for item in mcp_results.attractions)
    if query.tool_name == McpToolName.SEARCH_ACCOMMODATION_AREAS:
        return any(item.city == str(args["city"]) for item in mcp_results.accommodation_areas)
    if query.tool_name == McpToolName.SEARCH_LODGING_NEAR_PLACE:
        city = str(args["city"])
        anchor = str(args.get("anchor_place", ""))
        return any(item.city == city and (not anchor or _place_matches_any(anchor, [item.anchor_place])) for item in _mcp_lodging(mcp_results))
    return False


def _extract_query_result_from_results(query: McpQuery, mcp_results: McpResults) -> McpResults | None:
    args = query.args
    if query.tool_name == McpToolName.GET_WEATHER:
        query_date = _parse_date(str(args["date"]))
        items = [item for item in mcp_results.weather if item.city == str(args["city"]) and item.date == query_date]
        return McpResults(weather=items) if items else None
    if query.tool_name == McpToolName.GET_ROUTE_TIME:
        items = [item for item in mcp_results.routes if _route_matches_query(item, args)]
        return McpResults(routes=items) if items else None
    if query.tool_name == McpToolName.GET_ATTRACTION_DETAIL:
        query_date = _parse_date(str(args["date"]))
        query_city = str(args.get("city", ""))
        items = [
            item
            for item in mcp_results.attractions
            if item.date == query_date
            and (not query_city or item.city == query_city)
            and _place_matches_any(str(args["name"]), [item.name])
        ]
        return McpResults(attractions=items) if items else None
    if query.tool_name == McpToolName.SEARCH_ATTRACTIONS:
        items = [item for item in mcp_results.attractions if item.city == str(args["city"]) and item.date is None]
        return McpResults(attractions=items) if items else None
    if query.tool_name == McpToolName.SEARCH_ACCOMMODATION_AREAS:
        items = [item for item in mcp_results.accommodation_areas if item.city == str(args["city"])]
        return McpResults(accommodation_areas=items) if items else None
    if query.tool_name == McpToolName.SEARCH_LODGING_NEAR_PLACE:
        city = str(args["city"])
        anchor = str(args.get("anchor_place", ""))
        items = [
            item
            for item in _mcp_lodging(mcp_results)
            if item.city == city and (not anchor or _place_matches_any(anchor, [item.anchor_place]))
        ]
        return McpResults(lodging=items) if items else None
    return None


def _mcp_query_cache_key(query: McpQuery) -> str:
    return f"{query.tool_name.value}:{_stable_args_key(query.args)}"


def _stable_args_key(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)


def _update_skipped_existing_stats(state: TripState, skipped_count: int) -> dict[str, int]:
    previous = state.get("mcp_cache_stats", {})
    return {
        **previous,
        "skipped_existing_results": int(previous.get("skipped_existing_results", 0)) + skipped_count,
        "last_skipped_existing_results": skipped_count,
    }


def _execute_mock_mcp_query_plan(query_plan: McpQueryPlan, default_city: str) -> McpResults:
    collected = McpResults()
    for query in query_plan.queries:
        collected = _merge_mcp_results(collected, _execute_mock_mcp_query(query, default_city))
    return collected


def _execute_mock_mcp_query(query: McpQuery, default_city: str) -> McpResults:
    args = query.args
    if query.tool_name == McpToolName.GET_WEATHER:
        raw = mock_get_weather(city=str(args["city"]), date=str(args["date"]))
        return McpResults(
            weather=[
                WeatherResult(
                    city=str(raw["city"]),
                    date=_parse_date(str(raw["date"])),
                    condition=str(raw["condition"]),
                    warning=raw.get("warning"),
                )
            ]
        )
    if query.tool_name == McpToolName.SEARCH_ATTRACTIONS:
        raw_results = mock_search_attractions(city=str(args["city"]), preferences=list(args.get("preferences", [])))
        return McpResults(
            attractions=[
                AttractionResult(
                    name=str(item["name"]),
                    city=str(item.get("city", args["city"])),
                    category=PlaceCategory(str(item["category"])),
                    notes=str(item.get("match_reason", "")),
                )
                for item in raw_results
            ]
        )
    if query.tool_name == McpToolName.GET_ATTRACTION_DETAIL:
        query_date = _parse_date(str(args["date"]))
        raw = mock_get_attraction_detail(name=str(args["name"]), date=query_date.isoformat())
        return McpResults(
            attractions=[
                AttractionResult(
                    name=str(raw["name"]),
                    city=str(args.get("city", raw.get("city", default_city))),
                    category=PlaceCategory(str(raw["category"])),
                    date=query_date,
                    is_open=bool(raw["is_open"]),
                    opening_hours=str(raw["opening_hours"]),
                    ticket_price=float(raw["ticket_price"]),
                    recommended_duration_minutes=int(raw["recommended_duration_minutes"]),
                    notes=str(raw.get("notes", "")),
                )
            ]
        )
    if query.tool_name == McpToolName.GET_ROUTE_TIME:
        raw = mock_get_route_time(
            origin=str(args["origin"]),
            destination=str(args["destination"]),
            mode=str(args.get("mode", TransportMode.TAXI.value)),
        )
        return McpResults(
            routes=[
                RouteResult(
                    origin=str(raw["origin"]),
                    destination=str(raw["destination"]),
                    origin_city=str(args.get("origin_city", args.get("city", default_city))),
                    destination_city=str(args.get("destination_city", args.get("cityd", args.get("origin_city", default_city)))),
                    mode=TransportMode(str(raw["mode"])),
                    duration_minutes=int(raw["duration_minutes"]),
                    distance_km=float(raw["distance_km"]),
                )
            ]
        )
    if query.tool_name == McpToolName.SEARCH_ACCOMMODATION_AREAS:
        raw_results = mock_search_accommodation_areas(
            city=str(args["city"]),
            budget_level=str(args.get("budget_level", BudgetLevel.MEDIUM.value)),
            prefer_family_room=bool(args.get("prefer_family_room", False)),
        )
        return McpResults(
            accommodation_areas=[
                AccommodationAreaResult(
                    area_name=str(item["area_name"]),
                    city=str(item["city"]),
                    pros=list(item.get("pros", [])),
                    cons=list(item.get("cons", [])),
                    suitable_for=list(item.get("suitable_for", [])),
                    estimated_price_level=BudgetLevel(str(item.get("estimated_price_level", BudgetLevel.MEDIUM.value))),
                    notes=str(item.get("notes", "")),
                )
                for item in raw_results
            ]
        )
    if query.tool_name == McpToolName.SEARCH_LODGING_NEAR_PLACE:
        city = str(args["city"])
        anchor_place = str(args.get("anchor_place", city))
        return McpResults(
            lodging=[
                LodgingResult(
                    name=f"{anchor_place} Nearby Hotel",
                    city=city,
                    area=f"Near {anchor_place}",
                    address=f"{city} {anchor_place} area",
                    anchor_place=anchor_place,
                    distance_to_anchor_km=1.2,
                    duration_to_anchor_minutes=12,
                    estimated_price_level=BudgetLevel(str(args.get("budget_level", BudgetLevel.MEDIUM.value))),
                    notes="Mock lodging result tied to the stay anchor.",
                ),
                LodgingResult(
                    name=f"{city} Railway Station Hotel",
                    city=city,
                    area="Railway station area",
                    address=f"{city} railway station",
                    anchor_place=anchor_place,
                    distance_to_anchor_km=12,
                    duration_to_anchor_minutes=45,
                    estimated_price_level=BudgetLevel.LOW,
                    notes="Mock fallback lodging farther from the anchor.",
                ),
            ]
        )
    raise ValueError(f"Unsupported MCP tool: {query.tool_name}")


def _allow_mock_mcp_fallback(state: TripState) -> bool:
    explicit = state.get("allow_mock_mcp_fallback")
    if explicit is not None:
        return bool(explicit)
    return os.getenv("TRAVEL_AGENT_ALLOW_MOCK_MCP_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _format_exception_detail(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        details = "; ".join(_format_exception_detail(item) for item in exc.exceptions[:5])
        remaining = len(exc.exceptions) - 5
        suffix = f"; ... {remaining} more" if remaining > 0 else ""
        return f"{type(exc).__name__}: {details}{suffix}"
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _merge_mcp_results(existing: McpResults, incoming: McpResults) -> McpResults:
    weather = {(item.city, item.date): item for item in existing.weather}
    weather.update({(item.city, item.date): item for item in incoming.weather})
    attractions = {(item.name, item.city, item.date): item for item in existing.attractions}
    attractions.update({(item.name, item.city, item.date): item for item in incoming.attractions})
    routes = {_route_result_key(item): item for item in existing.routes}
    routes.update({_route_result_key(item): item for item in incoming.routes})
    areas = {(item.area_name, item.city): item for item in existing.accommodation_areas}
    areas.update({(item.area_name, item.city): item for item in incoming.accommodation_areas})
    lodging = {(item.name, item.city, item.anchor_place): item for item in _mcp_lodging(existing)}
    lodging.update({(item.name, item.city, item.anchor_place): item for item in _mcp_lodging(incoming)})
    return McpResults(
        weather=list(weather.values()),
        attractions=list(attractions.values()),
        routes=list(routes.values()),
        accommodation_areas=list(areas.values()),
        lodging=list(lodging.values()),
        errors=[*existing.errors, *incoming.errors],
    )


def _quality_gate_for_issues(issues: list[ValidationIssue]) -> PlanQualityGate:
    serious = [issue for issue in issues if issue.severity in {Severity.HIGH, Severity.CRITICAL}]
    return PlanQualityGate(
        can_finalize=not serious,
        blocking_issue_count=len(serious),
        max_severity=_max_severity(issues),
        reason=(
            "High or critical validation issues remain; final output is provisional."
            if serious
            else "No high or critical validation issues remain."
        ),
    )


def _max_severity(issues: list[ValidationIssue]) -> Severity | None:
    if not issues:
        return None
    order = {Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}
    return max((issue.severity for issue in issues), key=lambda severity: order[severity])


def _mcp_lodging(mcp_results: McpResults | object) -> list[LodgingResult]:
    value = getattr(mcp_results, "lodging", None)
    return value if isinstance(value, list) else []


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))


def _time_to_minutes(value: time | None) -> int:
    if value is None:
        return 0
    return value.hour * 60 + value.minute


def _minutes_to_time(minutes: int) -> time:
    minutes = max(0, min(23 * 60 + 59, int(minutes)))
    return time(minutes // 60, minutes % 60)


def _shift_time(value: time, minutes: int) -> time:
    return _minutes_to_time(_time_to_minutes(value) + minutes)


def _format_date_time(day_value: date | None, time_value: time | None) -> str:
    if day_value and time_value:
        return f"{day_value} {time_value.strftime('%H:%M')}"
    if day_value:
        return day_value.isoformat()
    if time_value:
        return time_value.strftime("%H:%M")
    return "unspecified"


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing} {note}".strip()


def _dedupe_and_sort_timeline(items: list[TimelineItem]) -> list[TimelineItem]:
    sorted_items = sorted(items, key=lambda item: (_time_to_minutes(item.start_time), item.sequence))
    deduped: list[TimelineItem] = []
    for item in sorted_items:
        original_duration = max(1, item.move.duration_minutes if item.move else item.duration_minutes)
        if deduped and item.start_time < deduped[-1].end_time:
            item.start_time = deduped[-1].end_time
            if _time_to_minutes(item.start_time) >= 23 * 60 + 59:
                continue
            item.end_time = _minutes_to_time(min(23 * 60 + 59, _time_to_minutes(item.start_time) + original_duration))
        if item.end_time <= item.start_time:
            continue
        if item.move:
            item.move.duration_minutes = item.duration_minutes
        if item.stay:
            item.stay.duration_minutes = item.duration_minutes
        deduped.append(item)
    _resequence_timeline(deduped)
    return deduped


def _resequence_timeline(items: list[TimelineItem]) -> None:
    for index, item in enumerate(sorted(items, key=lambda current: (_time_to_minutes(current.start_time), current.sequence)), start=1):
        item.sequence = index


def _covered_timeline_minutes(items: list[TimelineItem]) -> int:
    intervals = sorted((_time_to_minutes(item.start_time), _time_to_minutes(item.end_time)) for item in items)
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _segments_for_date(segments: list[TripSegment], current_date: date) -> list[TripSegment]:
    return [segment for segment in segments if segment.departure_date == current_date]


def _move_purpose_from_segment(segment_type: SegmentType) -> MovePurpose:
    if segment_type == SegmentType.OUTBOUND:
        return MovePurpose.OUTBOUND
    if segment_type == SegmentType.RETURN:
        return MovePurpose.RETURN
    if segment_type == SegmentType.INTERCITY:
        return MovePurpose.INTERCITY
    return MovePurpose.LOCAL


def _city_stay_for_date(city_route_plan: CityRoutePlan, current_date: date) -> CityStayPlan | None:
    return next((stay for stay in city_route_plan.stays if stay.start_date <= current_date <= stay.end_date), None)


def _stay_for_place(city_route_plan: CityRoutePlan, place: str) -> CityStayPlan | None:
    return next(
        (
            stay
            for stay in city_route_plan.stays
            if any(_place_matches_any(place, [anchor]) for anchor in stay.anchor_places)
        ),
        None,
    )


def _best_lodging_for_anchor(mcp_results: McpResults, city: str, anchor: str) -> LodgingResult | None:
    candidates = [
        lodging
        for lodging in _mcp_lodging(mcp_results)
        if lodging.city == city and (not anchor or _place_matches_any(anchor, [lodging.anchor_place]))
    ]
    if not candidates:
        candidates = [lodging for lodging in _mcp_lodging(mcp_results) if lodging.city == city]
    candidates = [item for item in candidates if not _lodging_conflicts_with_anchor(item, anchor)]
    scored = [
        item
        for item in candidates
        if (item.location and item.distance_to_anchor_km > 0)
        or (item.distance_to_anchor_km > 0 and item.duration_to_anchor_minutes > 0)
    ]
    if not scored:
        return None
    best = min(scored, key=lambda item: (item.duration_to_anchor_minutes or 9999, item.distance_to_anchor_km))
    if (best.duration_to_anchor_minutes and best.duration_to_anchor_minutes <= 60) or best.distance_to_anchor_km <= 30:
        return best
    return None


def _lodging_conflicts_with_anchor(lodging: LodgingResult, anchor: str) -> bool:
    normalized_anchor = _normalize_place_name(anchor)
    if "wutai" not in normalized_anchor:
        return False
    text = _normalize_place_name(f"{lodging.name} {lodging.area} {lodging.address} {lodging.city}")
    return "taiyuan" in text


def _preferred_accommodation_area(mcp_results: McpResults, city: str, anchor: str = "") -> str:
    area = next((item for item in mcp_results.accommodation_areas if item.city == city), None)
    if area and not _is_vague_or_remote_accommodation_area(area.area_name, city):
        return area.area_name
    return _unverified_lodging_label(city, anchor)


def _is_vague_or_remote_accommodation_area(area_name: str, city: str) -> bool:
    normalized = _normalize_place_name(area_name)
    normalized_city = _normalize_place_name(city)
    if not normalized:
        return True
    remote_tokens = {"station", "railwaystation", "airport"}
    if any(token in normalized for token in remote_tokens):
        return True
    lodging_poi_tokens = {"hotel", "inn", "guesthouse", "\u9152\u5e97", "\u6c11\u5bbf", "\u9a7f\u7ad9", "\u5bbe\u9986"}
    if any(token in normalized for token in lodging_poi_tokens):
        return True
    if normalized_city and normalized == normalized_city:
        return True
    return False


def _unverified_lodging_label(city: str, anchor: str = "") -> str:
    suffix = f" near {anchor}" if anchor else ""
    return f"{city} lodging unresolved{suffix}"


def _is_unverified_lodging_label(value: str) -> bool:
    normalized = _normalize_place_name(value)
    return "lodgingunresolved" in normalized or "unverifiedlodging" in normalized or "tobeconfirmed" in normalized


def _move_has_unverified_endpoint(move: MoveDetail) -> bool:
    return _is_unverified_lodging_label(move.origin) or _is_unverified_lodging_label(move.destination)


def _day_has_unreliable_lodging_routes(day: PlanDay) -> bool:
    if not day.accommodation_area:
        return False
    for item in day.timeline:
        if not item.move or item.move.purpose != MovePurpose.LOCAL:
            continue
        touches_lodging = _locations_equivalent(day.accommodation_area, item.move.origin) or _locations_equivalent(
            day.accommodation_area,
            item.move.destination,
        )
        if touches_lodging and (item.move.duration_minutes > 60 or item.move.distance_km > 30):
            return True
    return False


def _mark_lodging_unresolved(day: PlanDay, anchor: str) -> None:
    previous_lodging = day.accommodation_area or day.overnight_accommodation or ""
    label = _unverified_lodging_label(day.city, anchor)
    day.accommodation_area = label
    if day.overnight_accommodation:
        day.overnight_accommodation = label
    filtered: list[TimelineItem] = []
    for item in day.timeline:
        if item.stay and item.stay.purpose in {
            StayPurpose.SLEEP,
            StayPurpose.REST,
            StayPurpose.MEAL,
            StayPurpose.HOTEL_CHECKIN,
            StayPurpose.HOTEL_CHECKOUT,
        }:
            if not previous_lodging or _locations_equivalent(item.stay.place_name, previous_lodging):
                item.stay.place_name = label
                item.stay.notes = _append_note(item.stay.notes, "Lodging not verified by MCP.")
        if item.move and item.move.purpose == MovePurpose.LOCAL:
            touches_previous = previous_lodging and (
                _locations_equivalent(item.move.origin, previous_lodging)
                or _locations_equivalent(item.move.destination, previous_lodging)
            )
            touches_unverified = _is_unverified_lodging_label(item.move.origin) or _is_unverified_lodging_label(
                item.move.destination
            )
            if touches_previous or touches_unverified:
                continue
        filtered.append(item)
    day.timeline = filtered
    day.daily_notes = _append_note(day.daily_notes, "Lodging remains unresolved because MCP did not provide a nearby verified option.")
    _resequence_timeline(day.timeline)


def _cost_for_budget(level: BudgetLevel, low: float, medium: float, high: float) -> float:
    if level == BudgetLevel.LOW:
        return low
    if level == BudgetLevel.HIGH:
        return high
    return medium


def _route_duration_for(
    mcp_results: McpResults,
    origin: str,
    destination: str,
    mode: TransportMode,
    fallback: int,
) -> int:
    route = next(
        (
            item
            for item in mcp_results.routes
            if item.origin == origin and item.destination == destination and item.mode == mode
        ),
        None,
    )
    return route.duration_minutes if route else fallback


def _attraction_detail_for(
    mcp_results: McpResults,
    place: str,
    city: str,
    current_date: date | None,
) -> AttractionResult | None:
    dated = [
        item
        for item in mcp_results.attractions
        if (not city or item.city == city)
        and (current_date is None or item.date == current_date)
        and _place_matches_any(place, [item.name])
    ]
    if dated:
        return dated[0]
    return next(
        (
            item
            for item in mcp_results.attractions
            if (not city or item.city == city) and item.date is None and _place_matches_any(place, [item.name])
        ),
        None,
    )


def _category_for_place(place_name: str, mcp_results: McpResults, fallback: PlaceCategory) -> PlaceCategory:
    detail = _attraction_detail_for(mcp_results, place_name, "", None)
    return detail.category if detail else fallback


def _select_indoor_attraction(
    mcp_results: McpResults,
    city: str,
    excluded: set[str],
) -> AttractionResult | None:
    return next(
        (
            item
            for item in mcp_results.attractions
            if item.city == city and item.category == PlaceCategory.INDOOR and item.name not in excluded
        ),
        None,
    )


def _next_non_rain_day(plan: TripPlan, after_day: int, city: str, mcp_results: McpResults) -> PlanDay | None:
    for day in plan.days:
        if day.day <= after_day:
            continue
        if city and day.city != city:
            continue
        if not _is_bad_weather(mcp_results, day.city, day.date):
            return day
    return None


def _is_bad_weather(mcp_results: McpResults, city: str, current_date: date) -> bool:
    return any(
        item.city == city and item.date == current_date and item.condition in {"heavy rain", "storm", "snow"}
        for item in mcp_results.weather
    )


def _first_visit(day: PlanDay) -> TimelineItem | None:
    return next((item for item in day.timeline if item.stay and item.stay.purpose == StayPurpose.VISIT), None)


def _sync_accommodations_with_days(plan: TripPlan, mcp_results: McpResults) -> None:
    for accommodation in plan.accommodations:
        day = next((current for current in plan.days if current.city == accommodation.city and current.accommodation_area), None)
        if day is None or accommodation.hotel_name == day.accommodation_area:
            continue
        lodging = next((item for item in _mcp_lodging(mcp_results) if item.name == day.accommodation_area), None)
        accommodation.hotel_name = day.accommodation_area or accommodation.hotel_name
        if lodging:
            accommodation.area = lodging.area
            accommodation.address = lodging.address
            accommodation.location = lodging.location
            accommodation.reason = f"Changed near {lodging.anchor_place} during replan."


def _place_matches_any(required_place: str, planned_places: list[str]) -> bool:
    required = _normalize_place_name(required_place)
    for planned_place in planned_places:
        planned = _normalize_place_name(planned_place)
        if required == planned or required in planned or planned in required:
            return True
    return False


def _normalize_place_name(value: str) -> str:
    normalized = value.lower().strip()
    replacements = {
        " scenic area": "",
        " scenic spot": "",
        " tourist area": "",
        " province": "",
        " city": "",
        " ": "",
        "-": "",
        "_": "",
        "(": "",
        ")": "",
        "'": "",
        "（": "",
        "）": "",
        "风景名胜区": "",
        "风景区": "",
        "景区": "",
        "旅游区": "",
        "省": "",
        "市": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    aliases = {
        "wutaimountain": "wutai",
        "wutaishan": "wutai",
        "五台山": "wutai",
        "五台": "wutai",
        "xihu": "westlake",
        "西湖": "westlake",
        "humbleadministratorsgarden": "humbleadministratorgarden",
        "thehumbleadministratorsgarden": "humbleadministratorgarden",
        "zhuozhengyuan": "humbleadministratorgarden",
        "拙政园": "humbleadministratorgarden",
        "太原": "taiyuan",
        "忻州": "xinzhou",
        "台怀": "taihuai",
        "杭州": "hangzhou",
        "苏州": "suzhou",
        "山西": "shanxi",
        "浙江": "zhejiang",
        "江苏": "jiangsu",
    }
    for old, new in aliases.items():
        normalized = normalized.replace(old, new)
    return normalized


def _anchor_city_for_place(destination: str, place: str) -> str:
    text = _normalize_place_name(f"{destination}{place}")
    rules = [
        (("wutai", "taihuai", "xinzhou"), "Xinzhou"),
        (("westlake", "hangzhou"), "Hangzhou"),
        (("humbleadministratorgarden", "suzhou"), "Suzhou"),
        (("yungang", "云冈", "云岗"), "Datong"),
        (("xuankong", "悬空", "恒山"), "Datong"),
        (("pingyao", "平遥", "乔家"), "Jinzhong"),
        (("jinci", "晋祠", "taiyuan"), "Taiyuan"),
    ]
    for tokens, city in rules:
        if any(token in text for token in tokens):
            return city
    if "shanxi" in text:
        return "Taiyuan"
    if "zhejiang" in text:
        return "Hangzhou"
    if "jiangsu" in text:
        return "Suzhou"
    return destination


def _is_vague_route_endpoint(value: str) -> bool:
    normalized = _normalize_place_name(value)
    if _is_unverified_lodging_label(value):
        return True
    return normalized in {
        "shanxi",
        "zhejiang",
        "jiangsu",
        "sichuan",
        "四川",
        "yunnan",
        "云南",
        "destination",
        "citycenter",
        "中心",
        "市中心",
    }
