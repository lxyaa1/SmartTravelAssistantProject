from __future__ import annotations

from datetime import date, datetime, time, timedelta

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
    DayScheduleBlock,
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
    ParsedRequestOutput,
    PlanDay,
    PlanQualityGate,
    PlanCheckQueryPlannerInput,
    PlanCheckQueryPlannerOutput,
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
    RoutePlannerInput,
    RoutePlannerOutput,
    ScheduleBlockType,
    SegmentType,
    Severity,
    TransferLeg,
    TransportMode,
    TripSegment,
    TripPlan,
    TripRequest,
    ValidationIssue,
    ValidatorInput,
    ValidatorOutput,
    VisitSlot,
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
    """Parse raw input into the shared request schema."""
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
        must_visit=raw.get("must_visit", ["West Lake"]),
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
        "issues": [],
    }


def city_route_planner_node(state: TripState) -> TripState:
    """Create the city-level route skeleton before any detailed day planning."""
    agent_input = CityRoutePlannerInput(request=state["user_request"])
    request = agent_input.request

    if should_use_llm(state):
        city_route_plan = generate_city_route_plan_with_llm(request=request)
        city_route_plan = _normalize_city_route_plan(city_route_plan, request)
    else:
        city_route_plan = _build_city_route_plan(request)

    output = CityRoutePlannerOutput(city_route_plan=city_route_plan)
    return {**state, "city_route_plan": output.city_route_plan}


def preplan_query_planner_node(state: TripState) -> TripState:
    """Plan broad MCP queries needed before drafting an itinerary."""
    agent_input = PreplanQueryPlannerInput(
        request=state["user_request"],
        city_route_plan=state.get("city_route_plan"),
    )
    request = agent_input.request
    city_route_plan = agent_input.city_route_plan or _build_city_route_plan(request)
    stay_cities = [stay.city for stay in city_route_plan.stays] or [request.destination]
    queries: list[McpQuery] = [
    ]

    for city in dict.fromkeys(stay_cities):
        queries.extend(
            [
                McpQuery(
                    tool_name=McpToolName.SEARCH_ATTRACTIONS,
                    args={"city": city, "preferences": request.preferences},
                    purpose="Find optional attractions to fill time beyond must-visit places.",
                    stage=McpQueryStage.PREPLAN,
                ),
                McpQuery(
                    tool_name=McpToolName.SEARCH_ACCOMMODATION_AREAS,
                    args={
                        "city": city,
                        "budget_level": request.budget_level.value,
                        "prefer_family_room": request.accommodation.prefer_family_room if request.accommodation else False,
                    },
                    purpose="Find suitable fallback accommodation areas before planning daily routes.",
                    stage=McpQueryStage.PREPLAN,
                ),
            ]
        )

    for stay in city_route_plan.stays:
        for current_date in _date_range(stay.start_date, stay.end_date):
            queries.append(
                McpQuery(
                    tool_name=McpToolName.GET_WEATHER,
                    args={"city": stay.city, "date": current_date.isoformat()},
                    purpose="Check daily weather before assigning outdoor places.",
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
                purpose="Find lodging close to the main anchor instead of a generic city hotel.",
                stage=McpQueryStage.PREPLAN,
            )
        )

    for place in request.must_visit:
        target_stay = _stay_for_place(city_route_plan, place)
        queries.append(
            McpQuery(
                tool_name=McpToolName.GET_ATTRACTION_DETAIL,
                args={
                    "name": place,
                    "city": target_stay.city if target_stay else request.destination,
                    "date": target_stay.start_date.isoformat() if target_stay else request.start_date.isoformat(),
                },
                purpose="Check basic details for must-visit places before planning.",
                stage=McpQueryStage.PREPLAN,
            )
        )

    for segment in city_route_plan.segments:
        queries.append(_route_query_from_segment(segment, McpQueryStage.PREPLAN))

    output = PreplanQueryPlannerOutput(query_plan=McpQueryPlan(queries=_dedupe_mcp_queries(queries)))
    return {**state, "pending_mcp_queries": output.query_plan}


def draft_day_schedule_node(state: TripState) -> TripState:
    """Draft a detailed itinerary from route skeleton and pre-plan MCP data."""
    city_route_plan = state.get("city_route_plan") or _build_city_route_plan(state["user_request"])
    agent_input = DraftDayScheduleInput(
        request=state["user_request"],
        city_route_plan=city_route_plan,
        mcp_results=state.get("mcp_results", McpResults()),
    )
    request = agent_input.request
    mcp_results = agent_input.mcp_results

    if should_use_llm(state):
        plan = generate_initial_plan_with_llm(
            request=request,
            mcp_results=mcp_results,
            city_route_plan=agent_input.city_route_plan,
        )
    else:
        plan = _build_deterministic_trip_plan(
            request=request,
            city_route_plan=agent_input.city_route_plan,
            mcp_results=mcp_results,
        )

    _normalize_plan_after_generation(plan, request, agent_input.city_route_plan)
    output = DraftDayScheduleOutput(plan=plan)
    return {**state, "current_plan": output.plan}


def initial_plan_node(state: TripState) -> TripState:
    """Backward-compatible alias for older tests and scripts."""
    return draft_day_schedule_node(state)


def plan_check_query_planner_node(state: TripState) -> TripState:
    """Plan MCP queries needed to validate the current structured itinerary."""
    agent_input = PlanCheckQueryPlannerInput(plan=state["current_plan"])
    plan = agent_input.plan
    queries: list[McpQuery] = []

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
        if day.arrival_transfer:
            queries.append(_route_query_from_transfer(day.arrival_transfer, McpQueryStage.PLAN_CHECK))
        if day.start_transfer_to_first:
            queries.append(_route_query_from_transfer(day.start_transfer_to_first, McpQueryStage.PLAN_CHECK))
        if day.return_transfer_to_accommodation:
            queries.append(_route_query_from_transfer(day.return_transfer_to_accommodation, McpQueryStage.PLAN_CHECK))
        if day.departure_transfer:
            queries.append(_route_query_from_transfer(day.departure_transfer, McpQueryStage.PLAN_CHECK))
        for visit in day.visits:
            queries.append(
                McpQuery(
                    tool_name=McpToolName.GET_ATTRACTION_DETAIL,
                    args={"name": visit.place_name, "city": day.city, "date": day.date.isoformat()},
                    purpose="Verify attraction status for the planned date.",
                    stage=McpQueryStage.PLAN_CHECK,
                )
            )
            if visit.transport_to_next:
                queries.append(_route_query_from_transfer(visit.transport_to_next, McpQueryStage.PLAN_CHECK))
        for block in day.schedule_blocks:
            if block.transfer:
                queries.append(_route_query_from_transfer(block.transfer, McpQueryStage.PLAN_CHECK))

    output = PlanCheckQueryPlannerOutput(query_plan=McpQueryPlan(queries=_dedupe_mcp_queries(queries)))
    return {**state, "pending_mcp_queries": output.query_plan}


def _route_query_from_transfer(transfer: TransferLeg, stage: McpQueryStage) -> McpQuery:
    return McpQuery(
        tool_name=McpToolName.GET_ROUTE_TIME,
        args={
            "origin": transfer.origin,
            "destination": transfer.destination,
            "mode": transfer.mode.value,
        },
        purpose="Verify transfer duration for a planned movement.",
        stage=stage,
    )


def _route_query_from_segment(segment: TripSegment, stage: McpQueryStage) -> McpQuery:
    return McpQuery(
        tool_name=McpToolName.GET_ROUTE_TIME,
        args={
            "origin": segment.origin,
            "destination": segment.destination,
            "mode": segment.mode.value,
        },
        purpose=f"Verify {segment.segment_type.value} segment duration.",
        stage=stage,
    )


def _dedupe_mcp_queries(queries: list[McpQuery]) -> list[McpQuery]:
    seen: set[tuple[str, str]] = set()
    deduped: list[McpQuery] = []
    for query in queries:
        key = (query.tool_name.value, repr(sorted(query.args.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def collect_mcp_data_node(state: TripState) -> TripState:
    """Execute pending MCP queries and merge their results into state."""
    query_plan = state.get("pending_mcp_queries", McpQueryPlan())
    agent_input = DataCollectorInput(
        query_plan=query_plan,
        existing_results=state.get("mcp_results", McpResults()),
        default_city=state["user_request"].destination,
    )
    errors = list(state.get("mcp_errors", []))

    if should_use_amap_mcp(state):
        try:
            collected = execute_amap_mcp_query_plan(
                query_plan=agent_input.query_plan,
                default_city=agent_input.default_city,
            )
        except Exception as exc:
            errors.append(f"Amap MCP failed, fell back to mock MCP: {exc}")
            collected = _execute_mock_mcp_query_plan(agent_input.query_plan, default_city=agent_input.default_city)
    else:
        collected = _execute_mock_mcp_query_plan(agent_input.query_plan, default_city=agent_input.default_city)

    merged_results = _merge_mcp_results(agent_input.existing_results, collected)
    output = DataCollectorOutput(mcp_results=merged_results)
    return {
        **state,
        "mcp_results": output.mcp_results,
        "pending_mcp_queries": McpQueryPlan(),
        "mcp_errors": errors,
    }


def validate_plan_node(state: TripState) -> TripState:
    """Validate the detailed structured plan against collected MCP data."""
    agent_input = ValidatorInput(
        request=state["user_request"],
        plan=state["current_plan"],
        mcp_results=state["mcp_results"],
    )
    request = agent_input.request
    plan = agent_input.plan
    mcp_results = agent_input.mcp_results
    issues: list[ValidationIssue] = []
    route_by_key = {
        (item.origin, item.destination, item.mode): item
        for item in mcp_results.routes
    }
    _apply_route_results_to_plan(plan, route_by_key)

    if request.origin != request.destination and not any(
        segment.segment_type == SegmentType.RETURN for segment in plan.route_segments
    ):
        issues.append(
            ValidationIssue(
                issue_type=IssueType.MISSING_RETURN_TRANSFER,
                severity=Severity.CRITICAL,
                date=request.end_date,
                locations=[request.destination, request.origin],
                reason="The plan does not include an explicit return transfer to the origin.",
                suggested_action="Add a return route segment and a final-day departure transfer.",
            )
        )

    for segment in plan.route_segments:
        if _is_vague_route_endpoint(segment.origin) or _is_vague_route_endpoint(segment.destination):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.ROUTE_ENDPOINT_TOO_VAGUE,
                    severity=Severity.HIGH,
                    date=segment.departure_date,
                    locations=[segment.origin, segment.destination],
                    reason="Route endpoint is too broad for reliable travel-time validation.",
                    suggested_action="Use concrete cities, stations, hotels, or attractions instead of province-level endpoints.",
                )
            )
        if segment.segment_type == SegmentType.RETURN and segment.departure_date != request.end_date:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.TIME_CONFLICT,
                    severity=Severity.HIGH,
                    date=segment.departure_date,
                    locations=[segment.origin, segment.destination],
                    reason="Return segment is not scheduled on the final trip date.",
                    suggested_action="Move the return segment to the final day or adjust trip dates.",
                )
            )

    planned_places = [visit.place_name for day in plan.days for visit in day.visits]
    for place in request.must_visit:
        if not _place_matches_any(place, planned_places):
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.MISSING_MUST_VISIT,
                    severity=Severity.CRITICAL,
                    locations=[place],
                    reason=f"Required place {place} is not included.",
                    suggested_action="Add this place to one day of the itinerary.",
                )
            )

    weather_by_date = {(item.city, item.date): item for item in mcp_results.weather}
    attraction_by_name = {item.name: item for item in mcp_results.attractions}
    attraction_by_name_date = {
        (item.name, item.date): item
        for item in mcp_results.attractions
        if item.date is not None
    }
    lodging_by_name = {item.name: item for item in _mcp_lodging(mcp_results)}
    for day in plan.days:
        day_weather = weather_by_date.get((day.city, day.date))
        if day.arrival_transfer and day.visits:
            first_visit_start_minutes = day.visits[0].start_time.hour * 60 + day.visits[0].start_time.minute
            if day.arrival_transfer.estimated_duration_minutes > 180 and first_visit_start_minutes < 12 * 60:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.TIME_CONFLICT,
                        severity=Severity.HIGH,
                        day=day.day,
                        date=day.date,
                        locations=[day.arrival_transfer.origin, day.arrival_transfer.destination],
                        reason=(
                            f"Arrival transfer takes {day.arrival_transfer.estimated_duration_minutes} minutes, "
                            "but the first visit starts before noon."
                        ),
                        suggested_action="Reserve the first day for intercity travel or move visits to the afternoon.",
                    )
                )
        if day.day == len(plan.days) and request.origin != request.destination and day.departure_transfer is None:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.MISSING_RETURN_TRANSFER,
                    severity=Severity.CRITICAL,
                    day=day.day,
                    date=day.date,
                    locations=[day.city, request.origin],
                    reason="The final day has no departure_transfer back to the origin.",
                    suggested_action="Add a final-day return transfer and expose it in the daily schedule.",
                )
            )
        if day.schedule_blocks:
            covered_minutes = _covered_schedule_minutes(day.schedule_blocks)
            if covered_minutes < 18 * 60:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.INCOMPLETE_DAY_TIMELINE,
                        severity=Severity.MEDIUM,
                        day=day.day,
                        date=day.date,
                        locations=[day.city],
                        reason=f"Schedule blocks cover only {covered_minutes} minutes.",
                        suggested_action="Add sleep, meals, buffers, transfers, and rest blocks to cover the day more completely.",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.INCOMPLETE_DAY_TIMELINE,
                    severity=Severity.HIGH,
                    day=day.day,
                    date=day.date,
                    locations=[day.city],
                    reason="The day has no structured schedule blocks.",
                    suggested_action="Generate a 24-hour schedule with sleep, meals, transfers, visits, and rest.",
                )
            )
        for label, transfer in (
            ("start transfer", day.start_transfer_to_first),
            ("return transfer", day.return_transfer_to_accommodation),
            ("departure transfer", day.departure_transfer),
        ):
            if transfer and transfer.estimated_duration_minutes > 90:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.ROUTE_TOO_LONG,
                        severity=Severity.HIGH,
                        day=day.day,
                        date=day.date,
                        locations=[transfer.origin, transfer.destination],
                        reason=f"{label} takes {transfer.estimated_duration_minutes} minutes.",
                        suggested_action="Choose a closer hotel area, reorder the day, or split this place into another day.",
                    )
                )
        if day.accommodation_area and day.visits:
            lodging = lodging_by_name.get(day.accommodation_area)
            for anchor in {day.visits[0].place_name, day.visits[-1].place_name}:
                route = next(
                    (
                        item
                        for item in mcp_results.routes
                        if item.mode == TransportMode.TAXI
                        and (
                            (item.origin == day.accommodation_area and item.destination == anchor)
                            or (item.origin == anchor and item.destination == day.accommodation_area)
                        )
                    ),
                    None,
                )
                duration = route.duration_minutes if route else lodging.duration_to_anchor_minutes if lodging else 0
                if duration > 45:
                    issues.append(
                        ValidationIssue(
                            issue_type=IssueType.LODGING_TOO_FAR,
                            severity=Severity.HIGH,
                            day=day.day,
                            date=day.date,
                            locations=[day.accommodation_area, anchor],
                            reason=f"Lodging-to-anchor transfer takes {duration} minutes.",
                            suggested_action="Pick lodging near the main anchor place or move this attraction to a closer overnight base.",
                        )
                    )
        for visit in day.visits:
            attraction = attraction_by_name_date.get((visit.place_name, day.date)) or attraction_by_name.get(
                visit.place_name
            )
            if attraction and not attraction.is_open:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.ATTRACTION_CLOSED,
                        severity=Severity.HIGH,
                        day=day.day,
                        date=day.date,
                        locations=[visit.place_name],
                        reason=f"{visit.place_name} is closed on this date.",
                        suggested_action="Move this visit to another day or replace it with an indoor alternative.",
                    )
                )
            if day_weather and day_weather.condition == "heavy rain" and visit.category == PlaceCategory.OUTDOOR:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.BAD_WEATHER,
                        severity=Severity.HIGH,
                        day=day.day,
                        date=day.date,
                        locations=[visit.place_name],
                        reason=f"{visit.place_name} is outdoors and the weather is heavy rain.",
                        suggested_action="Swap this outdoor visit with an indoor visit on another day.",
                    )
                )
        for origin, destination in zip(day.visits, day.visits[1:]):
            mode = origin.transport_to_next.mode if origin.transport_to_next else TransportMode.TAXI
            route = route_by_key.get((origin.place_name, destination.place_name, mode))
            if route and route.duration_minutes > 90:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.ROUTE_TOO_LONG,
                        severity=Severity.HIGH,
                        day=day.day,
                        date=day.date,
                        locations=[route.origin, route.destination],
                        reason=f"Travel time is {route.duration_minutes} minutes.",
                        suggested_action="Reorder nearby places or split them across different days.",
                    )
                )

    quality_gate = _quality_gate_for_issues(issues)
    plan.quality_gate = quality_gate
    output = ValidatorOutput(issues=issues, quality_gate=quality_gate)
    return {**state, "issues": output.issues, "current_plan": plan}


def repair_strategy_planner_node(state: TripState) -> TripState:
    """Decide whether the graph should repair again, finalize, or expose infeasibility."""
    agent_input = RepairStrategyPlannerInput(
        issues=state.get("issues", []),
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 3),
    )
    serious = [
        issue
        for issue in agent_input.issues
        if issue.severity in {Severity.HIGH, Severity.CRITICAL}
    ]
    if not serious:
        strategy = RepairStrategy(action=RepairAction.FINALIZE, reason="No high or critical issues remain.")
    elif agent_input.iteration >= agent_input.max_iterations:
        strategy = RepairStrategy(
            action=RepairAction.INFEASIBLE,
            reason="Maximum replanning iterations reached while high or critical issues remain.",
            target_issue_types=list({issue.issue_type for issue in serious}),
        )
    else:
        strategy = RepairStrategy(
            action=RepairAction.REPLAN,
            reason="High or critical issues remain and replanning budget is available.",
            target_issue_types=list({issue.issue_type for issue in serious}),
        )

    output = RepairStrategyPlannerOutput(repair_strategy=strategy)
    return {**state, "repair_strategy": output.repair_strategy}


def replan_node(state: TripState) -> TripState:
    """Apply a small deterministic fix so the loop can be tested."""
    agent_input = ReplannerInput(
        request=state["user_request"],
        current_plan=state["current_plan"],
        issues=state.get("issues", []),
        mcp_results=state["mcp_results"],
        iteration=state.get("iteration", 0),
    )
    request = agent_input.request
    current_plan = agent_input.current_plan
    plan_versions = [*state.get("plan_versions", []), current_plan]

    if should_use_llm(state):
        next_plan = replan_with_llm(
            request=request,
            current_plan=current_plan,
            issues=agent_input.issues,
            mcp_results=agent_input.mcp_results,
        )
        _normalize_plan_after_generation(
            next_plan,
            request,
            state.get("city_route_plan") or _build_city_route_plan(request),
        )
        output = ReplannerOutput(plan=next_plan, addressed_issues=agent_input.issues)
        return {
            **state,
            "current_plan": output.plan,
            "plan_versions": plan_versions,
            "iteration": state.get("iteration", 0) + 1,
            "issues": [],
        }

    next_plan = current_plan.model_copy(deep=True)
    issue_types = {issue.issue_type for issue in agent_input.issues}
    bad_weather_locations = {
        location
        for issue in agent_input.issues
        if issue.issue_type == IssueType.BAD_WEATHER
        for location in issue.locations
    }
    closed_occurrences = {
        (issue.day, location)
        for issue in agent_input.issues
        if issue.issue_type == IssueType.ATTRACTION_CLOSED
        for location in issue.locations
    }
    long_route_occurrences = {
        (issue.day, location)
        for issue in agent_input.issues
        if issue.issue_type == IssueType.ROUTE_TOO_LONG
        for location in issue.locations
    }

    for day in next_plan.days:
        day_changed = False
        for visit in day.visits:
            if (
                IssueType.BAD_WEATHER in issue_types
                and day.day == 1
                and visit.place_name in bad_weather_locations
                and visit.category == PlaceCategory.OUTDOOR
            ):
                visit.place_name = f"{day.city} Art Gallery"
                visit.category = PlaceCategory.INDOOR
                visit.notes = "Replanned from outdoor visit because of mock heavy rain."
                day_changed = True
            if (day.day, visit.place_name) in closed_occurrences:
                visit.place_name = f"{day.city} Tea House"
                visit.category = PlaceCategory.FOOD
                visit.notes = "Replanned from a closed attraction."
                day_changed = True
            if (
                IssueType.ROUTE_TOO_LONG in issue_types
                and visit.transport_to_next
                and (
                    (day.day, visit.place_name) in long_route_occurrences
                    or (day.day, visit.transport_to_next.destination) in long_route_occurrences
                )
            ):
                visit.transport_to_next.mode = TransportMode.TRANSIT
                day_changed = True
            if (day.day, visit.place_name) in long_route_occurrences and visit.place_name not in request.must_visit:
                visit.place_name = f"{day.city} Tea House"
                visit.category = PlaceCategory.FOOD
                visit.notes = "Replanned from a long transfer."
                day_changed = True
        if day_changed:
            day.daily_notes = "This day was adjusted by the mock replanner."

    planned_places = {visit.place_name for day in next_plan.days for visit in day.visits}
    for required_place in request.must_visit:
        if required_place not in planned_places:
            target_day = next((day for day in next_plan.days if day.day > 1), next_plan.days[0])
            if target_day.visits:
                target_day.visits[0].place_name = required_place
                target_day.visits[0].category = PlaceCategory.OUTDOOR
                target_day.visits[0].notes = "Moved here to avoid mock heavy rain on the first day."

    _apply_structural_repairs(
        plan=next_plan,
        request=request,
        issues=agent_input.issues,
        mcp_results=agent_input.mcp_results,
        city_route_plan=state.get("city_route_plan") or _build_city_route_plan(request),
    )
    next_plan.assumptions = [*next_plan.assumptions, "One mock replanning pass was applied."]
    _normalize_plan_after_generation(
        next_plan,
        request,
        state.get("city_route_plan") or _build_city_route_plan(request),
    )
    output = ReplannerOutput(plan=next_plan, addressed_issues=agent_input.issues)
    return {
        **state,
        "current_plan": output.plan,
        "plan_versions": plan_versions,
        "iteration": state.get("iteration", 0) + 1,
        "issues": [],
    }


def final_writer_node(state: TripState) -> TripState:
    """Render a compact final response from the structured plan."""
    agent_input = FinalWriterInput(
        plan=state["current_plan"],
        unresolved_issues=state.get("issues", []),
    )
    plan = agent_input.plan
    lines = [f"# {plan.title}", ""]
    if not plan.quality_gate.can_finalize:
        lines.extend(
            [
                "> This plan still has blocking validation issues and should be treated as provisional.",
                f"> {plan.quality_gate.reason}",
                "",
            ]
        )
    if plan.route_segments:
        lines.extend(["## Route", ""])
        for segment in plan.route_segments:
            departure = _format_segment_time(segment.departure_date, segment.departure_time)
            arrival = _format_segment_time(segment.arrival_date, segment.arrival_time)
            label = segment.segment_type.value
            lines.append(
                f"- {label}: {segment.origin} -> {segment.destination}, {segment.mode.value}, "
                f"{departure} to {arrival}, {segment.estimated_duration_minutes} min"
            )
            if segment.station_or_terminal or segment.booking_notes:
                detail = "; ".join(item for item in [segment.station_or_terminal, segment.booking_notes] if item)
                lines.append(f"  - {detail}")
        lines.append("")
    if plan.accommodations:
        lines.extend(["## Accommodation", ""])
        for stay in plan.accommodations:
            lines.append(
                f"- {stay.check_in_date} to {stay.check_out_date}: {stay.hotel_name}, "
                f"{stay.city} {stay.area}".strip()
            )
            if stay.reason:
                lines.append(f"  - {stay.reason}")
        lines.append("")
    for day in plan.days:
        lines.append(f"## Day {day.day} - {day.date} - {day.city}")
        if day.schedule_blocks:
            for block in day.schedule_blocks:
                detail = block.title
                if block.transfer:
                    detail = f"{detail}: {_format_transfer(block.transfer)}"
                lines.append(
                    f"- {block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')} "
                    f"{block.block_type.value}: {detail}"
                )
        else:
            if day.arrival_transfer:
                lines.append(f"- Arrival: {_format_transfer(day.arrival_transfer)}")
            if day.start_transfer_to_first:
                lines.append(f"- Start transfer: {_format_transfer(day.start_transfer_to_first)}")
            for visit in day.visits:
                lines.append(
                    f"- {visit.start_time.strftime('%H:%M')}-{visit.end_time.strftime('%H:%M')} "
                    f"{visit.place_name} ({visit.category.value})"
                )
            if day.return_transfer_to_accommodation:
                lines.append(f"- Return transfer: {_format_transfer(day.return_transfer_to_accommodation)}")
            if day.departure_transfer:
                lines.append(f"- Departure: {_format_transfer(day.departure_transfer)}")
        if day.daily_notes:
            lines.append(f"  Note: {day.daily_notes}")
        lines.append("")

    final_plan = FinalPlan(content="\n".join(lines).strip(), unresolved_issues=agent_input.unresolved_issues)
    output = FinalWriterOutput(final_plan=final_plan)
    return {**state, "final_plan": output.final_plan}


def _format_transfer(transfer: TransferLeg) -> str:
    return (
        f"{transfer.origin} -> {transfer.destination}, {transfer.mode.value}, "
        f"{transfer.estimated_duration_minutes} min, {transfer.estimated_distance_km:.1f} km"
    )


def _format_segment_time(segment_date: date | None, segment_time: time | None) -> str:
    if segment_date and segment_time:
        return f"{segment_date.isoformat()} {segment_time.strftime('%H:%M')}"
    if segment_date:
        return segment_date.isoformat()
    if segment_time:
        return segment_time.strftime("%H:%M")
    return "time TBD"


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    days = (end - start).days + 1
    return [start + timedelta(days=offset) for offset in range(days)]


def _build_city_route_plan(request: TripRequest) -> CityRoutePlan:
    days = _date_range(request.start_date, request.end_date)
    anchors = request.must_visit or [request.destination]
    city_to_anchors: dict[str, list[str]] = {}
    for anchor in anchors:
        city_to_anchors.setdefault(_anchor_city_for_place(request.destination, anchor), []).append(anchor)

    if not city_to_anchors:
        city_to_anchors[request.destination] = [request.destination]

    stays: list[CityStayPlan] = []
    segments: list[TripSegment] = []
    city_items = list(city_to_anchors.items())
    stay_count = len(city_items)
    current_day_index = 0
    for index, (city, city_anchors) in enumerate(city_items, start=1):
        remaining_days = len(days) - current_day_index
        remaining_stays = stay_count - index + 1
        span = max(1, remaining_days // remaining_stays)
        if index == stay_count:
            span = remaining_days
        stay_dates = days[current_day_index : current_day_index + span]
        current_day_index += span
        start_date = stay_dates[0]
        end_date = stay_dates[-1]
        lodging_anchor = city_anchors[0] if city_anchors else city
        stays.append(
            CityStayPlan(
                sequence=index,
                city=city,
                start_date=start_date,
                end_date=end_date,
                anchor_places=city_anchors,
                lodging_anchor=lodging_anchor,
                notes="Concrete stay inferred from destination and must-visit anchors.",
            )
        )

    sequence = 1
    first_stay = stays[0]
    if request.origin != first_stay.city:
        segments.append(
            _make_trip_segment(
                sequence=sequence,
                segment_type=SegmentType.OUTBOUND,
                origin=request.origin,
                destination=first_stay.city,
                origin_city=request.origin,
                destination_city=first_stay.city,
                departure_date=request.start_date,
                departure_time=time(8, 0),
                arrival_date=request.start_date,
                arrival_time=time(12, 0),
                notes="Outbound transfer is required before first-day local planning.",
            )
        )
        sequence += 1

    for previous, current in zip(stays, stays[1:]):
        segments.append(
            _make_trip_segment(
                sequence=sequence,
                segment_type=SegmentType.INTERCITY,
                origin=previous.city,
                destination=current.city,
                origin_city=previous.city,
                destination_city=current.city,
                departure_date=current.start_date,
                departure_time=time(8, 30),
                arrival_date=current.start_date,
                arrival_time=time(12, 0),
                notes="Intercity transfer between inferred stay cities.",
            )
        )
        sequence += 1

    last_stay = stays[-1]
    if request.origin != last_stay.city:
        segments.append(
            _make_trip_segment(
                sequence=sequence,
                segment_type=SegmentType.RETURN,
                origin=last_stay.city,
                destination=request.origin,
                origin_city=last_stay.city,
                destination_city=request.origin,
                departure_date=request.end_date,
                departure_time=time(17, 0),
                arrival_date=request.end_date,
                arrival_time=time(21, 0),
                notes="Return transfer is required so the trip ends back at the origin.",
            )
        )

    return CityRoutePlan(
        origin=request.origin,
        destination=request.destination,
        stays=stays,
        segments=segments,
        notes=[
            "City route skeleton is generated before day planning.",
            "Province or region destinations are converted to concrete stay cities where possible.",
        ],
    )


def _normalize_city_route_plan(city_route_plan: CityRoutePlan, request: TripRequest) -> CityRoutePlan:
    if not city_route_plan.stays:
        return _build_city_route_plan(request)

    normalized = city_route_plan.model_copy(deep=True)
    normalized.origin = request.origin
    normalized.destination = request.destination
    if not any(segment.segment_type == SegmentType.RETURN for segment in normalized.segments):
        last_stay = normalized.stays[-1]
        if request.origin != last_stay.city:
            normalized.segments.append(
                _make_trip_segment(
                    sequence=len(normalized.segments) + 1,
                    segment_type=SegmentType.RETURN,
                    origin=last_stay.city,
                    destination=request.origin,
                    origin_city=last_stay.city,
                    destination_city=request.origin,
                    departure_date=request.end_date,
                    departure_time=time(17, 0),
                    arrival_date=request.end_date,
                    arrival_time=time(21, 0),
                    notes="Return segment added by normalization.",
                )
            )
    return normalized


def _make_trip_segment(
    sequence: int,
    segment_type: SegmentType,
    origin: str,
    destination: str,
    origin_city: str,
    destination_city: str,
    departure_date: date,
    departure_time: time,
    arrival_date: date,
    arrival_time: time,
    notes: str,
) -> TripSegment:
    duration = _minutes_between(departure_time, arrival_time)
    return TripSegment(
        sequence=sequence,
        segment_type=segment_type,
        origin=origin,
        destination=destination,
        origin_city=origin_city,
        destination_city=destination_city,
        mode=TransportMode.TRAIN,
        departure_date=departure_date,
        departure_time=departure_time,
        arrival_date=arrival_date,
        arrival_time=arrival_time,
        estimated_duration_minutes=duration,
        estimated_distance_km=0,
        estimated_cost=150 if segment_type != SegmentType.INTERCITY else 80,
        station_or_terminal="To be confirmed from railway/flight booking source",
        train_or_flight_number="",
        booking_notes="Specific train/flight number is not available from Amap MCP; verify in a ticketing source.",
        notes=notes,
    )


def _build_deterministic_trip_plan(
    request: TripRequest,
    city_route_plan: CityRoutePlan,
    mcp_results: McpResults,
) -> TripPlan:
    accommodations = [
        _accommodation_for_stay(stay, request, mcp_results)
        for stay in city_route_plan.stays
    ]
    accommodation_by_city = {stay.city: accommodation for stay, accommodation in zip(city_route_plan.stays, accommodations)}
    plan_days: list[PlanDay] = []
    days = _date_range(request.start_date, request.end_date)

    for index, current_date in enumerate(days, start=1):
        stay = _stay_for_date(city_route_plan, current_date) or city_route_plan.stays[-1]
        accommodation = accommodation_by_city.get(stay.city)
        visits = _visits_for_day(
            request=request,
            stay=stay,
            current_date=current_date,
            day_index=index,
            mcp_results=mcp_results,
        )
        arrival_segment = _segment_for_date(city_route_plan, current_date, {SegmentType.OUTBOUND, SegmentType.INTERCITY})
        return_segment = _segment_for_date(city_route_plan, current_date, {SegmentType.RETURN})
        hotel_name = accommodation.hotel_name if accommodation else _preferred_accommodation_area(mcp_results, stay.city)

        day = PlanDay(
            day=index,
            date=current_date,
            city=stay.city,
            visits=visits,
            accommodation_area=hotel_name,
            overnight_accommodation=hotel_name if current_date < request.end_date else None,
            arrival_transfer=_segment_to_transfer(arrival_segment) if arrival_segment else None,
            start_transfer_to_first=(
                TransferLeg(
                    origin=hotel_name,
                    destination=visits[0].place_name,
                    mode=TransportMode.TAXI,
                    estimated_duration_minutes=15 if accommodation else 30,
                    estimated_distance_km=2 if accommodation else 8,
                    estimated_cost=20,
                    notes="Hotel-to-first-stop transfer; validated by MCP in plan-check stage.",
                )
                if hotel_name and visits
                else None
            ),
            return_transfer_to_accommodation=(
                TransferLeg(
                    origin=visits[-1].place_name,
                    destination=hotel_name,
                    mode=TransportMode.TAXI,
                    estimated_duration_minutes=15 if accommodation else 30,
                    estimated_distance_km=2 if accommodation else 8,
                    estimated_cost=20,
                    notes="Last-stop-to-hotel transfer; validated by MCP in plan-check stage.",
                )
                if hotel_name and visits and current_date < request.end_date
                else None
            ),
            departure_transfer=_segment_to_transfer(return_segment) if return_segment else None,
            daily_notes="Drafted from route skeleton, lodging anchors, weather, and attraction candidates.",
        )
        day.schedule_blocks = _build_day_schedule_blocks(day, request)
        plan_days.append(day)

    plan = TripPlan(
        title=f"{request.origin} to {request.destination} structured itinerary",
        origin=request.origin,
        destination=request.destination,
        route_segments=[segment.model_copy(deep=True) for segment in city_route_plan.segments],
        accommodations=accommodations,
        days=plan_days,
        assumptions=[
            "Amap MCP can validate map routes and POIs but cannot provide confirmed train numbers or ticket inventory.",
            (
                f"Travel party has {request.travelers.total_people} people and "
                f"{request.accommodation.bed_count if request.accommodation else request.travelers.bed_count} "
                "required beds for accommodation planning."
            ),
        ],
    )
    return plan


def _normalize_plan_after_generation(plan: TripPlan, request: TripRequest, city_route_plan: CityRoutePlan) -> None:
    if not plan.route_segments:
        plan.route_segments = [segment.model_copy(deep=True) for segment in city_route_plan.segments]
    if not plan.accommodations:
        plan.accommodations = [
            AccommodationStay(
                hotel_name=stay.lodging_anchor or f"{stay.city} central lodging",
                city=stay.city,
                area=stay.lodging_anchor or stay.city,
                check_in_date=stay.start_date,
                check_out_date=stay.end_date,
                bed_count=request.accommodation.bed_count if request.accommodation else request.travelers.bed_count,
                reason="Fallback accommodation generated from route skeleton.",
                nearby_anchor_places=stay.anchor_places,
            )
            for stay in city_route_plan.stays
        ]
    _ensure_plan_day_transfers(plan, request)
    _sync_transfer_names(plan)
    for day in plan.days:
        if not day.schedule_blocks:
            day.schedule_blocks = _build_day_schedule_blocks(day, request)
    _recalculate_plan_totals(plan)


def _apply_structural_repairs(
    plan: TripPlan,
    request: TripRequest,
    issues: list[ValidationIssue],
    mcp_results: McpResults,
    city_route_plan: CityRoutePlan,
) -> None:
    issue_types = {issue.issue_type for issue in issues}

    if IssueType.MISSING_RETURN_TRANSFER in issue_types:
        if not any(segment.segment_type == SegmentType.RETURN for segment in plan.route_segments):
            last_stay = city_route_plan.stays[-1] if city_route_plan.stays else CityStayPlan(
                sequence=1,
                city=request.destination,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            plan.route_segments.append(
                _make_trip_segment(
                    sequence=len(plan.route_segments) + 1,
                    segment_type=SegmentType.RETURN,
                    origin=last_stay.city,
                    destination=request.origin,
                    origin_city=last_stay.city,
                    destination_city=request.origin,
                    departure_date=request.end_date,
                    departure_time=time(17, 0),
                    arrival_date=request.end_date,
                    arrival_time=time(21, 0),
                    notes="Added by structural repair because return transfer was missing.",
                )
            )
        if plan.days:
            return_segment = next(
                (segment for segment in plan.route_segments if segment.segment_type == SegmentType.RETURN),
                None,
            )
            if return_segment:
                plan.days[-1].departure_transfer = _segment_to_transfer(return_segment)

    if IssueType.LODGING_TOO_FAR in issue_types or IssueType.ROUTE_TOO_LONG in issue_types:
        for day in plan.days:
            stay = _stay_for_date(city_route_plan, day.date)
            if not stay:
                continue
            lodging = _preferred_lodging(mcp_results, stay)
            if not lodging:
                continue
            day.accommodation_area = lodging.name
            if day.overnight_accommodation:
                day.overnight_accommodation = lodging.name
            if day.start_transfer_to_first and day.visits:
                day.start_transfer_to_first.origin = lodging.name
                day.start_transfer_to_first.destination = day.visits[0].place_name
                day.start_transfer_to_first.estimated_duration_minutes = max(10, lodging.duration_to_anchor_minutes or 15)
                day.start_transfer_to_first.estimated_distance_km = lodging.distance_to_anchor_km or 2
                day.start_transfer_to_first.notes = "Changed to lodging near the stay anchor during structural repair."
            if day.return_transfer_to_accommodation and day.visits:
                day.return_transfer_to_accommodation.origin = day.visits[-1].place_name
                day.return_transfer_to_accommodation.destination = lodging.name
                day.return_transfer_to_accommodation.estimated_duration_minutes = max(10, lodging.duration_to_anchor_minutes or 15)
                day.return_transfer_to_accommodation.estimated_distance_km = lodging.distance_to_anchor_km or 2
                day.return_transfer_to_accommodation.notes = "Changed to lodging near the stay anchor during structural repair."
            existing_stay = next((item for item in plan.accommodations if item.city == lodging.city), None)
            if existing_stay:
                existing_stay.hotel_name = lodging.name
                existing_stay.area = lodging.area
                existing_stay.address = lodging.address
                existing_stay.location = lodging.location
                existing_stay.reason = f"Repaired to stay near {lodging.anchor_place}."

    if IssueType.INCOMPLETE_DAY_TIMELINE in issue_types or IssueType.MISSING_RETURN_TRANSFER in issue_types:
        for day in plan.days:
            day.schedule_blocks = _build_day_schedule_blocks(day, request)


def _accommodation_for_stay(
    stay: CityStayPlan,
    request: TripRequest,
    mcp_results: McpResults,
) -> AccommodationStay:
    lodging = _preferred_lodging(mcp_results, stay)
    bed_count = request.accommodation.bed_count if request.accommodation else request.travelers.bed_count
    if lodging:
        return AccommodationStay(
            hotel_name=lodging.name,
            city=lodging.city,
            area=lodging.area,
            address=lodging.address,
            location=lodging.location,
            check_in_date=stay.start_date,
            check_out_date=stay.end_date,
            bed_count=bed_count,
            room_count=request.accommodation.room_count if request.accommodation else None,
            reason=f"Chosen because it is near {lodging.anchor_place or stay.lodging_anchor}.",
            nearby_anchor_places=stay.anchor_places or [stay.lodging_anchor],
            estimated_cost_per_night=350,
            notes=lodging.notes,
        )

    area = _preferred_accommodation_area(mcp_results, stay.city)
    return AccommodationStay(
        hotel_name=area,
        city=stay.city,
        area=area,
        check_in_date=stay.start_date,
        check_out_date=stay.end_date,
        bed_count=bed_count,
        room_count=request.accommodation.room_count if request.accommodation else None,
        reason=f"Fallback lodging area near {stay.lodging_anchor or stay.city}.",
        nearby_anchor_places=stay.anchor_places or [stay.lodging_anchor],
        estimated_cost_per_night=300,
        notes="Specific lodging was not available from MCP; using an area-level fallback.",
    )


def _preferred_lodging(mcp_results: McpResults, stay: CityStayPlan) -> LodgingResult | None:
    candidates = [
        item
        for item in _mcp_lodging(mcp_results)
        if item.city == stay.city
        and (
            not stay.lodging_anchor
            or _place_matches_any(stay.lodging_anchor, [item.anchor_place, item.name, item.area])
        )
    ]
    if not candidates:
        candidates = [item for item in _mcp_lodging(mcp_results) if item.city == stay.city]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.duration_to_anchor_minutes or 999, item.distance_to_anchor_km or 999))[0]


def _visits_for_day(
    request: TripRequest,
    stay: CityStayPlan,
    current_date: date,
    day_index: int,
    mcp_results: McpResults,
) -> list[VisitSlot]:
    must_visit_for_stay = [
        place
        for place in request.must_visit
        if _anchor_city_for_place(request.destination, place) == stay.city
    ]
    used: set[str] = set()
    first_place = (
        must_visit_for_stay[0]
        if current_date == stay.start_date and must_visit_for_stay
        else _select_candidate_attraction(
            mcp_results=mcp_results,
            city=stay.city,
            prefer_indoor=_is_heavy_rain(mcp_results, stay.city, current_date),
            excluded=used,
            fallback=f"{stay.city} Art Gallery" if _is_heavy_rain(mcp_results, stay.city, current_date) else stay.lodging_anchor or stay.city,
        )
    )
    used.add(first_place)
    second_place = _select_candidate_attraction(
        mcp_results=mcp_results,
        city=stay.city,
        prefer_indoor=False,
        excluded=used,
        fallback=f"{stay.city} Old Street",
    )
    first_start = time(14, 0) if day_index == 1 else time(9, 30)
    first_end = time(16, 0) if day_index == 1 else time(11, 30)
    second_start = time(16, 30) if day_index == 1 else time(14, 30)
    second_end = time(18, 0) if day_index == 1 else time(16, 30)

    return [
        VisitSlot(
            sequence=1,
            place_name=first_place,
            city=stay.city,
            category=_category_for_place(first_place, mcp_results, PlaceCategory.OUTDOOR),
            start_time=first_start,
            end_time=first_end,
            visit_duration_minutes=_minutes_between(first_start, first_end),
            transport_to_next=TransferLeg(
                origin=first_place,
                destination=second_place,
                mode=TransportMode.TAXI,
                estimated_duration_minutes=25,
                estimated_distance_km=6,
                estimated_cost=25,
                notes="Between-visit transfer; validated by MCP in plan-check stage.",
            ),
            estimated_cost=50,
        ),
        VisitSlot(
            sequence=2,
            place_name=second_place,
            city=stay.city,
            category=_category_for_place(second_place, mcp_results, PlaceCategory.CULTURE),
            start_time=second_start,
            end_time=second_end,
            visit_duration_minutes=_minutes_between(second_start, second_end),
            estimated_cost=30,
        ),
    ]


def _build_day_schedule_blocks(day: PlanDay, request: TripRequest) -> list[DayScheduleBlock]:
    blocks: list[DayScheduleBlock] = []
    sequence = 1

    def add(
        block_type: ScheduleBlockType,
        start: time,
        end: time,
        title: str,
        place_name: str | None = None,
        transfer: TransferLeg | None = None,
        cost: float = 0,
        notes: str = "",
    ) -> None:
        nonlocal sequence
        blocks.append(
            DayScheduleBlock(
                sequence=sequence,
                block_type=block_type,
                start_time=start,
                end_time=end,
                title=title,
                city=day.city,
                place_name=place_name,
                transfer=transfer,
                estimated_cost=cost,
                notes=notes,
            )
        )
        sequence += 1

    add(ScheduleBlockType.SLEEP, time(0, 0), time(7, 30), "Sleep")
    add(ScheduleBlockType.BREAKFAST, time(7, 30), time(8, 15), "Breakfast")
    if day.arrival_transfer:
        add(
            ScheduleBlockType.INTERCITY_TRANSFER,
            time(8, 15),
            time(12, 15),
            f"Travel from {day.arrival_transfer.origin} to {day.arrival_transfer.destination}",
            transfer=day.arrival_transfer,
            cost=day.arrival_transfer.estimated_cost,
        )
        add(ScheduleBlockType.LUNCH, time(12, 15), time(13, 15), "Lunch after arrival")
        add(ScheduleBlockType.HOTEL_CHECKIN, time(13, 15), time(14, 0), "Hotel check-in", day.accommodation_area)
    else:
        add(ScheduleBlockType.FREE_TIME, time(8, 15), time(9, 0), "Morning buffer")

    if day.start_transfer_to_first and day.visits:
        first_visit = day.visits[0]
        transfer_start = _shift_time(first_visit.start_time, -day.start_transfer_to_first.estimated_duration_minutes)
        add(
            ScheduleBlockType.LOCAL_TRANSFER,
            transfer_start,
            first_visit.start_time,
            f"Transfer to {first_visit.place_name}",
            first_visit.place_name,
            day.start_transfer_to_first,
            day.start_transfer_to_first.estimated_cost,
        )

    for visit in day.visits:
        add(
            ScheduleBlockType.VISIT,
            visit.start_time,
            visit.end_time,
            f"Visit {visit.place_name}",
            visit.place_name,
            cost=visit.estimated_cost,
            notes=visit.notes,
        )
        if visit.transport_to_next:
            transfer_start = visit.end_time
            transfer_end = _shift_time(visit.end_time, visit.transport_to_next.estimated_duration_minutes)
            add(
                ScheduleBlockType.LOCAL_TRANSFER,
                transfer_start,
                transfer_end,
                f"Transfer to {visit.transport_to_next.destination}",
                visit.transport_to_next.destination,
                visit.transport_to_next,
                visit.transport_to_next.estimated_cost,
            )

    add(ScheduleBlockType.DINNER, time(18, 30), time(19, 30), "Dinner")
    if day.return_transfer_to_accommodation:
        add(
            ScheduleBlockType.LOCAL_TRANSFER,
            time(19, 30),
            time(20, 0),
            f"Return to {day.return_transfer_to_accommodation.destination}",
            day.return_transfer_to_accommodation.destination,
            day.return_transfer_to_accommodation,
            day.return_transfer_to_accommodation.estimated_cost,
        )
        add(ScheduleBlockType.REST, time(20, 0), time(22, 30), "Evening rest")
    if day.departure_transfer:
        add(ScheduleBlockType.HOTEL_CHECKOUT, time(16, 0), time(16, 30), "Hotel checkout", day.accommodation_area)
        add(
            ScheduleBlockType.INTERCITY_TRANSFER,
            time(17, 0),
            time(21, 0),
            f"Return from {day.departure_transfer.origin} to {day.departure_transfer.destination}",
            transfer=day.departure_transfer,
            cost=day.departure_transfer.estimated_cost,
        )
        add(ScheduleBlockType.REST, time(21, 0), time(22, 30), "Arrival buffer")
    add(ScheduleBlockType.SLEEP, time(22, 30), time(23, 59), "Sleep")
    return _dedupe_overlapping_blocks(blocks)


def _dedupe_overlapping_blocks(blocks: list[DayScheduleBlock]) -> list[DayScheduleBlock]:
    ordered = sorted(blocks, key=lambda block: (block.start_time, block.sequence))
    deduped: list[DayScheduleBlock] = []
    for block in ordered:
        if deduped and block.start_time < deduped[-1].end_time and block.block_type not in {
            ScheduleBlockType.VISIT,
            ScheduleBlockType.INTERCITY_TRANSFER,
        }:
            continue
        block.sequence = len(deduped) + 1
        deduped.append(block)
    return deduped


def _stay_for_date(city_route_plan: CityRoutePlan, current_date: date) -> CityStayPlan | None:
    for stay in city_route_plan.stays:
        if stay.start_date <= current_date <= stay.end_date:
            return stay
    return None


def _stay_for_place(city_route_plan: CityRoutePlan, place: str) -> CityStayPlan | None:
    for stay in city_route_plan.stays:
        if _place_matches_any(place, stay.anchor_places):
            return stay
    return city_route_plan.stays[0] if city_route_plan.stays else None


def _segment_for_date(
    city_route_plan: CityRoutePlan,
    current_date: date,
    segment_types: set[SegmentType],
) -> TripSegment | None:
    for segment in city_route_plan.segments:
        if segment.segment_type in segment_types and segment.departure_date == current_date:
            return segment
    return None


def _segment_to_transfer(segment: TripSegment) -> TransferLeg:
    return TransferLeg(
        origin=segment.origin,
        destination=segment.destination,
        mode=segment.mode,
        estimated_duration_minutes=segment.estimated_duration_minutes,
        estimated_distance_km=segment.estimated_distance_km,
        estimated_cost=segment.estimated_cost,
        notes=segment.notes or segment.booking_notes,
    )


def _anchor_city_for_place(destination: str, place: str) -> str:
    text = _normalize_place_name(f"{destination}{place}")
    rules = [
        (("wutai", "五台"), "忻州"),
        (("yungang", "云冈", "云岗"), "大同"),
        (("xuankong", "悬空", "恒山"), "大同"),
        (("pingyao", "平遥", "乔家"), "晋中"),
        (("taiyuan", "太原", "晋祠"), "太原"),
    ]
    for tokens, city in rules:
        if any(token in text for token in tokens):
            return city
    if "shanxi" in text or "山西" in text:
        return "太原"
    return destination


def _minutes_between(start: time, end: time) -> int:
    return max(0, end.hour * 60 + end.minute - start.hour * 60 - start.minute)


def _shift_time(value: time, minutes: int) -> time:
    total = max(0, min(23 * 60 + 59, value.hour * 60 + value.minute + minutes))
    return time(total // 60, total % 60)


def _execute_mock_mcp_query_plan(query_plan: McpQueryPlan, default_city: str) -> McpResults:
    collected = McpResults()
    for query in query_plan.queries:
        collected = _merge_mcp_results(
            collected,
            _execute_mock_mcp_query(query, default_city=default_city),
        )
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
        raw_results = mock_search_attractions(
            city=str(args["city"]),
            preferences=list(args.get("preferences", [])),
        )
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
                    notes="Mock lodging result intentionally tied to the stay anchor.",
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


def _merge_mcp_results(existing: McpResults, incoming: McpResults) -> McpResults:
    weather = {(item.city, item.date): item for item in existing.weather}
    weather.update({(item.city, item.date): item for item in incoming.weather})

    attractions = {(item.name, item.city, item.date): item for item in existing.attractions}
    attractions.update({(item.name, item.city, item.date): item for item in incoming.attractions})

    routes = {(item.origin, item.destination, item.mode): item for item in existing.routes}
    routes.update({(item.origin, item.destination, item.mode): item for item in incoming.routes})

    accommodation_areas = {
        (item.area_name, item.city): item
        for item in existing.accommodation_areas
    }
    accommodation_areas.update(
        {
            (item.area_name, item.city): item
            for item in incoming.accommodation_areas
        }
    )

    lodging = {(item.name, item.city, item.anchor_place): item for item in _mcp_lodging(existing)}
    lodging.update({(item.name, item.city, item.anchor_place): item for item in _mcp_lodging(incoming)})

    return McpResults(
        weather=list(weather.values()),
        attractions=list(attractions.values()),
        routes=list(routes.values()),
        accommodation_areas=list(accommodation_areas.values()),
        lodging=list(lodging.values()),
    )


def _choose_must_visit_day(request: TripRequest, mcp_results: McpResults, days: list[date]) -> int:
    if not request.must_visit:
        return 1
    for index, current_date in enumerate(days, start=1):
        if not _is_heavy_rain(mcp_results, request.destination, current_date):
            return index
    return 1


def _is_heavy_rain(mcp_results: McpResults, city: str, current_date: date) -> bool:
    return any(
        item.city == city and item.date == current_date and item.condition == "heavy rain"
        for item in mcp_results.weather
    )


def _select_candidate_attraction(
    mcp_results: McpResults,
    city: str,
    prefer_indoor: bool,
    excluded: set[str],
    fallback: str,
) -> str:
    candidates = [
        item
        for item in mcp_results.attractions
        if item.city == city and item.name not in excluded
    ]
    if prefer_indoor:
        indoor = next((item for item in candidates if item.category == PlaceCategory.INDOOR), None)
        if indoor:
            return indoor.name
    if candidates:
        return candidates[0].name
    return fallback


def _category_for_place(place_name: str, mcp_results: McpResults, fallback: PlaceCategory) -> PlaceCategory:
    for attraction in mcp_results.attractions:
        if attraction.name == place_name:
            return attraction.category
    return fallback


def _preferred_accommodation_area(mcp_results: McpResults, city: str) -> str:
    for area in mcp_results.accommodation_areas:
        if area.city == city:
            return area.area_name
    return f"{city} city center"


def _place_matches_any(required_place: str, planned_places: list[str]) -> bool:
    required = _normalize_place_name(required_place)
    for planned_place in planned_places:
        planned = _normalize_place_name(planned_place)
        if required == planned or required in planned or planned in required:
            return True
    return False


def _normalize_place_name_legacy(value: str) -> str:
    normalized = value.lower().strip()
    replacements = {
        "岗": "冈",
        "風": "风",
        "臺": "台",
        "臺": "台",
        "景区": "",
        "风景名胜区": "",
        "风景区": "",
        "旅游区": "",
        " ": "",
        "-": "",
        "(": "",
        ")": "",
        "（": "",
        "）": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _ensure_plan_day_transfers(plan: TripPlan, request: TripRequest) -> None:
    if not plan.route_segments:
        city_route_plan = _build_city_route_plan(request)
        plan.route_segments = [segment.model_copy(deep=True) for segment in city_route_plan.segments]

    for day in plan.days:
        if day.day == 1 and request.origin != request.destination and day.arrival_transfer is None:
            arrival_segment = next(
                (
                    segment
                    for segment in plan.route_segments
                    if segment.segment_type in {SegmentType.OUTBOUND, SegmentType.INTERCITY}
                    and segment.departure_date == day.date
                ),
                None,
            )
            day.arrival_transfer = (
                _segment_to_transfer(arrival_segment)
                if arrival_segment
                else TransferLeg(
                    origin=request.origin,
                    destination=day.city,
                    mode=TransportMode.TRAIN,
                    estimated_duration_minutes=240,
                    estimated_distance_km=0,
                    estimated_cost=150,
                    notes="Added automatically so intercity arrival is explicit.",
                )
            )

        if day.day == len(plan.days) and request.origin != request.destination and day.departure_transfer is None:
            return_segment = next(
                (
                    segment
                    for segment in plan.route_segments
                    if segment.segment_type == SegmentType.RETURN and segment.departure_date == day.date
                ),
                None,
            )
            if return_segment:
                day.departure_transfer = _segment_to_transfer(return_segment)

        if not day.accommodation_area or not day.visits:
            continue

        if day.start_transfer_to_first is None:
            day.start_transfer_to_first = TransferLeg(
                origin=day.accommodation_area,
                destination=day.visits[0].place_name,
                mode=TransportMode.TAXI,
                estimated_duration_minutes=30,
                estimated_distance_km=0,
                estimated_cost=30,
                notes="Added automatically for hotel-to-first-stop validation.",
            )
        if day.return_transfer_to_accommodation is None:
            day.return_transfer_to_accommodation = TransferLeg(
                origin=day.visits[-1].place_name,
                destination=day.accommodation_area,
                mode=TransportMode.TAXI,
                estimated_duration_minutes=30,
                estimated_distance_km=0,
                estimated_cost=30,
                notes="Added automatically for last-stop-to-hotel validation.",
            )


def _sync_transfer_names(plan: TripPlan) -> None:
    for day in plan.days:
        if day.start_transfer_to_first and day.visits:
            day.start_transfer_to_first.destination = day.visits[0].place_name
        if day.return_transfer_to_accommodation and day.visits:
            day.return_transfer_to_accommodation.origin = day.visits[-1].place_name
        for origin, destination in zip(day.visits, day.visits[1:]):
            if origin.transport_to_next:
                origin.transport_to_next.origin = origin.place_name
                origin.transport_to_next.destination = destination.place_name
        if day.visits:
            day.visits[-1].transport_to_next = None


def _recalculate_plan_totals(plan: TripPlan) -> None:
    for day in plan.days:
        day.total_visit_minutes = sum(visit.visit_duration_minutes for visit in day.visits)
        day.total_transport_minutes = sum(
            visit.transport_to_next.estimated_duration_minutes
            for visit in day.visits
            if visit.transport_to_next
        )
        for transfer in (day.arrival_transfer, day.start_transfer_to_first, day.return_transfer_to_accommodation):
            if transfer:
                day.total_transport_minutes += transfer.estimated_duration_minutes
        if day.departure_transfer:
            day.total_transport_minutes += day.departure_transfer.estimated_duration_minutes
        day.sleep_minutes = sum(
            _minutes_between(block.start_time, block.end_time)
            for block in day.schedule_blocks
            if block.block_type == ScheduleBlockType.SLEEP
        )
        day.estimated_cost = sum(
            visit.estimated_cost + (visit.transport_to_next.estimated_cost if visit.transport_to_next else 0)
            for visit in day.visits
        )
        for transfer in (day.arrival_transfer, day.start_transfer_to_first, day.return_transfer_to_accommodation):
            if transfer:
                day.estimated_cost += transfer.estimated_cost
        if day.departure_transfer:
            day.estimated_cost += day.departure_transfer.estimated_cost
    plan.total_estimated_cost = sum(day.estimated_cost for day in plan.days)
    if plan.accommodations:
        plan.total_estimated_cost += sum(
            stay.estimated_cost_per_night
            * max(1, (stay.check_out_date - stay.check_in_date).days)
            for stay in plan.accommodations
        )


def _apply_route_results_to_plan(
    plan: TripPlan,
    route_by_key: dict[tuple[str, str, TransportMode], RouteResult],
) -> None:
    for segment in plan.route_segments:
        route = route_by_key.get((segment.origin, segment.destination, segment.mode))
        if not route:
            continue
        segment.estimated_duration_minutes = route.duration_minutes
        segment.estimated_distance_km = route.distance_km
        segment.notes = "Updated from MCP route data."
    for day in plan.days:
        for transfer in _day_transfers(day):
            route = route_by_key.get((transfer.origin, transfer.destination, transfer.mode))
            if not route:
                continue
            transfer.estimated_duration_minutes = route.duration_minutes
            transfer.estimated_distance_km = route.distance_km
            transfer.notes = "Updated from MCP route data."
    _recalculate_plan_totals(plan)


def _day_transfers(day: PlanDay) -> list[TransferLeg]:
    transfers: list[TransferLeg] = []
    for transfer in (
        day.arrival_transfer,
        day.start_transfer_to_first,
        day.return_transfer_to_accommodation,
        day.departure_transfer,
    ):
        if transfer:
            transfers.append(transfer)
    for visit in day.visits:
        if visit.transport_to_next:
            transfers.append(visit.transport_to_next)
    for block in day.schedule_blocks:
        if block.transfer:
            transfers.append(block.transfer)
    return transfers


def _quality_gate_for_issues(issues: list[ValidationIssue]) -> PlanQualityGate:
    serious = [issue for issue in issues if issue.severity in {Severity.HIGH, Severity.CRITICAL}]
    max_severity = _max_severity(issues)
    return PlanQualityGate(
        can_finalize=not serious,
        blocking_issue_count=len(serious),
        max_severity=max_severity,
        reason=(
            "High or critical validation issues remain; final output should be treated as infeasible or provisional."
            if serious
            else "No high or critical validation issues remain."
        ),
    )


def _mcp_lodging(mcp_results: McpResults | object) -> list[LodgingResult]:
    value = getattr(mcp_results, "lodging", None)
    if value is None:
        return []
    return value if isinstance(value, list) else []


def _max_severity(issues: list[ValidationIssue]) -> Severity | None:
    if not issues:
        return None
    order = {
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }
    return max((issue.severity for issue in issues), key=lambda severity: order[severity])


def _covered_schedule_minutes(blocks: list[DayScheduleBlock]) -> int:
    intervals = sorted(
        (
            block.start_time.hour * 60 + block.start_time.minute,
            block.end_time.hour * 60 + block.end_time.minute,
        )
        for block in blocks
    )
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _is_vague_route_endpoint(value: str) -> bool:
    normalized = _normalize_place_name(value)
    vague_tokens = {
        "shanxi",
        "山西",
        "zhejiang",
        "浙江",
        "jiangsu",
        "江苏",
        "sichuan",
        "四川",
        "yunnan",
        "云南",
        "目的地",
        "市中心",
        "中心",
        "citycenter",
    }
    return normalized in vague_tokens


def _normalize_place_name(value: str) -> str:
    normalized = value.lower().strip()
    replacements = {
        "云岗": "云冈",
        "風": "风",
        "臺": "台",
        "台懷": "台怀",
        "风景名胜区": "",
        "风景旅游区": "",
        "风景区": "",
        "景区": "",
        "旅游区": "",
        "省": "",
        "市": "",
        " ": "",
        "-": "",
        "(": "",
        ")": "",
        "（": "",
        "）": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized
