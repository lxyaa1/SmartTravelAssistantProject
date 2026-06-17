from __future__ import annotations

from datetime import date, datetime, time, timedelta

from backend.agents.llm import generate_initial_plan_with_llm, replan_with_llm, should_use_llm
from backend.graph.state import TripState
from backend.schemas.trip import (
    AccommodationAreaResult,
    AttractionResult,
    BudgetLevel,
    DataCollectorInput,
    DataCollectorOutput,
    FinalPlan,
    FinalWriterInput,
    FinalWriterOutput,
    IssueType,
    McpQuery,
    McpQueryPlan,
    McpQueryStage,
    McpResults,
    McpToolName,
    ParsedRequestOutput,
    PlanDay,
    PlanCheckQueryPlannerInput,
    PlanCheckQueryPlannerOutput,
    PlaceCategory,
    PreplanQueryPlannerInput,
    PreplanQueryPlannerOutput,
    ReplannerInput,
    ReplannerOutput,
    RouteResult,
    RoutePlannerInput,
    RoutePlannerOutput,
    Severity,
    TransferLeg,
    TransportMode,
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
        "issues": [],
    }


def preplan_query_planner_node(state: TripState) -> TripState:
    """Plan broad MCP queries needed before drafting an itinerary."""
    agent_input = PreplanQueryPlannerInput(request=state["user_request"])
    request = agent_input.request
    queries: list[McpQuery] = [
        McpQuery(
            tool_name=McpToolName.SEARCH_ATTRACTIONS,
            args={"city": request.destination, "preferences": request.preferences},
            purpose="Find optional attractions to fill time beyond must-visit places.",
            stage=McpQueryStage.PREPLAN,
        ),
        McpQuery(
            tool_name=McpToolName.SEARCH_ACCOMMODATION_AREAS,
            args={
                "city": request.destination,
                "budget_level": request.budget_level.value,
                "prefer_family_room": request.accommodation.prefer_family_room if request.accommodation else False,
            },
            purpose="Find suitable accommodation areas before planning daily routes.",
            stage=McpQueryStage.PREPLAN,
        ),
    ]

    for current_date in _date_range(request.start_date, request.end_date):
        queries.append(
            McpQuery(
                tool_name=McpToolName.GET_WEATHER,
                args={"city": request.destination, "date": current_date.isoformat()},
                purpose="Check daily weather before assigning outdoor places.",
                stage=McpQueryStage.PREPLAN,
            )
        )

    for place in request.must_visit:
        queries.append(
            McpQuery(
                tool_name=McpToolName.GET_ATTRACTION_DETAIL,
                args={"name": place, "date": request.start_date.isoformat()},
                purpose="Check basic details for must-visit places before planning.",
                stage=McpQueryStage.PREPLAN,
            )
        )

    output = PreplanQueryPlannerOutput(query_plan=McpQueryPlan(queries=queries))
    return {**state, "pending_mcp_queries": output.query_plan}


def initial_plan_node(state: TripState) -> TripState:
    """Create a simple initial plan. Replace this with an LLM planner later."""
    agent_input = RoutePlannerInput(
        request=state["user_request"],
        mcp_results=state.get("mcp_results", McpResults()),
    )
    request = agent_input.request
    mcp_results = agent_input.mcp_results
    if should_use_llm(state):
        plan = generate_initial_plan_with_llm(request=request, mcp_results=mcp_results)
        output = RoutePlannerOutput(plan=plan)
        return {**state, "current_plan": output.plan}

    days = _date_range(request.start_date, request.end_date)
    plan_days: list[PlanDay] = []
    must_visit_day = _choose_must_visit_day(request, mcp_results, days)
    accommodation_area = _preferred_accommodation_area(mcp_results, request.destination)

    for index, current_date in enumerate(days, start=1):
        must_visit_name = request.must_visit[0] if request.must_visit else f"{request.destination} Museum"
        if index == must_visit_day:
            first_place = must_visit_name
        else:
            first_place = _select_candidate_attraction(
                mcp_results=mcp_results,
                city=request.destination,
                prefer_indoor=_is_heavy_rain(mcp_results, request.destination, current_date),
                excluded={must_visit_name},
                fallback=f"{request.destination} Old Street",
            )
        first_category = _category_for_place(first_place, mcp_results, PlaceCategory.OUTDOOR)
        visits = [
            VisitSlot(
                sequence=1,
                place_name=first_place,
                city=request.destination,
                category=first_category,
                start_time=time(9, 0),
                end_time=time(11, 30),
                visit_duration_minutes=150,
                transport_to_next=TransferLeg(
                    origin=first_place,
                    destination=f"{request.destination} Museum",
                    mode=TransportMode.TAXI,
                    estimated_duration_minutes=35,
                    estimated_distance_km=8,
                    estimated_cost=25,
                ),
                estimated_cost=50,
            ),
            VisitSlot(
                sequence=2,
                place_name=f"{request.destination} Museum",
                city=request.destination,
                category=PlaceCategory.INDOOR,
                start_time=time(14, 0),
                end_time=time(16, 30),
                visit_duration_minutes=150,
                transport_to_next=None,
                estimated_cost=30,
            ),
        ]
        plan_days.append(
            PlanDay(
                day=index,
                date=current_date,
                city=request.destination,
                visits=visits,
                accommodation_area=accommodation_area,
                total_visit_minutes=sum(visit.visit_duration_minutes for visit in visits),
                total_transport_minutes=sum(
                    visit.transport_to_next.estimated_duration_minutes
                    for visit in visits
                    if visit.transport_to_next
                ),
                estimated_cost=sum(
                    visit.estimated_cost
                    + (visit.transport_to_next.estimated_cost if visit.transport_to_next else 0)
                    for visit in visits
                ),
            )
        )

    plan = TripPlan(
        title=f"{request.origin} to {request.destination} mock itinerary",
        origin=request.origin,
        destination=request.destination,
        days=plan_days,
        total_estimated_cost=sum(day.estimated_cost for day in plan_days),
        assumptions=[
            "Mock itinerary generated without real API calls.",
            (
                f"Travel party has {request.travelers.total_people} people and "
                f"{request.accommodation.bed_count if request.accommodation else request.travelers.bed_count} "
                "required beds for accommodation planning."
            ),
        ],
    )
    output = RoutePlannerOutput(plan=plan)
    return {**state, "current_plan": output.plan}


def plan_check_query_planner_node(state: TripState) -> TripState:
    """Plan MCP queries needed to validate the current structured itinerary."""
    agent_input = PlanCheckQueryPlannerInput(plan=state["current_plan"])
    plan = agent_input.plan
    queries: list[McpQuery] = []

    for day in plan.days:
        queries.append(
            McpQuery(
                tool_name=McpToolName.GET_WEATHER,
                args={"city": day.city, "date": day.date.isoformat()},
                purpose="Verify weather for the planned day.",
                stage=McpQueryStage.PLAN_CHECK,
            )
        )
        for visit in day.visits:
            queries.append(
                McpQuery(
                    tool_name=McpToolName.GET_ATTRACTION_DETAIL,
                    args={"name": visit.place_name, "date": day.date.isoformat()},
                    purpose="Verify attraction status for the planned date.",
                    stage=McpQueryStage.PLAN_CHECK,
                )
            )
        for origin, destination in zip(day.visits, day.visits[1:]):
            mode = origin.transport_to_next.mode.value if origin.transport_to_next else TransportMode.TAXI.value
            queries.append(
                McpQuery(
                    tool_name=McpToolName.GET_ROUTE_TIME,
                    args={
                        "origin": origin.place_name,
                        "destination": destination.place_name,
                        "mode": mode,
                    },
                    purpose="Verify transfer duration between consecutive planned places.",
                    stage=McpQueryStage.PLAN_CHECK,
                )
            )

    output = PlanCheckQueryPlannerOutput(query_plan=McpQueryPlan(queries=queries))
    return {**state, "pending_mcp_queries": output.query_plan}


def collect_mcp_data_node(state: TripState) -> TripState:
    """Execute pending MCP queries and merge their results into state."""
    agent_input = DataCollectorInput(
        query_plan=state.get("pending_mcp_queries", McpQueryPlan()),
        existing_results=state.get("mcp_results", McpResults()),
        default_city=state["user_request"].destination,
    )
    collected = McpResults()

    for query in agent_input.query_plan.queries:
        collected = _merge_mcp_results(
            collected,
            _execute_mock_mcp_query(query, default_city=agent_input.default_city),
        )

    merged_results = _merge_mcp_results(agent_input.existing_results, collected)
    output = DataCollectorOutput(mcp_results=merged_results)
    return {**state, "mcp_results": output.mcp_results, "pending_mcp_queries": McpQueryPlan()}


def validate_plan_node(state: TripState) -> TripState:
    """Find feasibility issues from mock data."""
    agent_input = ValidatorInput(
        request=state["user_request"],
        plan=state["current_plan"],
        mcp_results=state["mcp_results"],
    )
    request = agent_input.request
    plan = agent_input.plan
    mcp_results = agent_input.mcp_results
    issues: list[ValidationIssue] = []

    planned_places = {visit.place_name for day in plan.days for visit in day.visits}
    for place in request.must_visit:
        if place not in planned_places:
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
    route_by_key = {
        (item.origin, item.destination, item.mode): item
        for item in mcp_results.routes
    }

    for day in plan.days:
        day_weather = weather_by_date.get((day.city, day.date))
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

    output = ValidatorOutput(issues=issues)
    return {**state, "issues": output.issues}


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
        _sync_transfer_names(next_plan)
        _recalculate_plan_totals(next_plan)
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

    next_plan.assumptions = [*next_plan.assumptions, "One mock replanning pass was applied."]
    _sync_transfer_names(next_plan)
    _recalculate_plan_totals(next_plan)
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
    for day in plan.days:
        lines.append(f"## Day {day.day} - {day.date} - {day.city}")
        for visit in day.visits:
            lines.append(
                f"- {visit.start_time.strftime('%H:%M')}-{visit.end_time.strftime('%H:%M')} "
                f"{visit.place_name} ({visit.category.value})"
            )
        if day.daily_notes:
            lines.append(f"  Note: {day.daily_notes}")
        lines.append("")

    final_plan = FinalPlan(content="\n".join(lines).strip(), unresolved_issues=agent_input.unresolved_issues)
    output = FinalWriterOutput(final_plan=final_plan)
    return {**state, "final_plan": output.final_plan}


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    days = (end - start).days + 1
    return [start + timedelta(days=offset) for offset in range(days)]


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
                    city=str(raw.get("city", default_city)),
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

    return McpResults(
        weather=list(weather.values()),
        attractions=list(attractions.values()),
        routes=list(routes.values()),
        accommodation_areas=list(accommodation_areas.values()),
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


def _sync_transfer_names(plan: TripPlan) -> None:
    for day in plan.days:
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
        day.estimated_cost = sum(
            visit.estimated_cost + (visit.transport_to_next.estimated_cost if visit.transport_to_next else 0)
            for visit in day.visits
        )
    plan.total_estimated_cost = sum(day.estimated_cost for day in plan.days)
