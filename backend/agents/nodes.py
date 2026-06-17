from __future__ import annotations

from datetime import date, datetime, time, timedelta

from backend.graph.state import TripState
from backend.schemas.trip import (
    AttractionResult,
    BudgetLevel,
    DataCollectorInput,
    DataCollectorOutput,
    FinalPlan,
    FinalWriterInput,
    FinalWriterOutput,
    IssueType,
    McpResults,
    ParsedRequestOutput,
    PlanDay,
    PlaceCategory,
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
        "issues": [],
    }


def initial_plan_node(state: TripState) -> TripState:
    """Create a simple initial plan. Replace this with an LLM planner later."""
    agent_input = RoutePlannerInput(request=state["user_request"])
    request = agent_input.request
    days = _date_range(request.start_date, request.end_date)
    plan_days: list[PlanDay] = []

    for index, current_date in enumerate(days, start=1):
        must_visit_name = request.must_visit[0] if request.must_visit else f"{request.destination} Museum"
        first_place = must_visit_name if index == 1 else f"{request.destination} Old Street"
        visits = [
            VisitSlot(
                sequence=1,
                place_name=first_place,
                city=request.destination,
                category=PlaceCategory.OUTDOOR if index == 1 else PlaceCategory.CULTURE,
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
                accommodation_area=f"{request.destination} city center",
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


def collect_mcp_data_node(state: TripState) -> TripState:
    """Collect mock external data for the current plan."""
    agent_input = DataCollectorInput(plan=state["current_plan"])
    plan = agent_input.plan
    weather: list[WeatherResult] = []
    attractions: list[AttractionResult] = []
    routes: list[RouteResult] = []

    for day in plan.days:
        weather.append(
            WeatherResult(
                city=day.city,
                date=day.date,
                condition="heavy rain" if day.day == 1 else "cloudy",
                warning="Outdoor plans may be affected." if day.day == 1 else None,
            )
        )

        for visit in day.visits:
            attractions.append(
                AttractionResult(
                    name=visit.place_name,
                    city=visit.city,
                    category=visit.category,
                    is_open=not (visit.place_name.endswith("Museum") and day.day == 2),
                    opening_hours="Closed" if visit.place_name.endswith("Museum") and day.day == 2 else "09:00-18:00",
                    ticket_price=visit.estimated_cost,
                    recommended_duration_minutes=120,
                )
            )

        for origin, destination in zip(day.visits, day.visits[1:]):
            long_first_day_route = day.day == 1 and (
                origin.place_name == "West Lake" or destination.place_name == "West Lake"
            )
            routes.append(
                RouteResult(
                    origin=origin.place_name,
                    destination=destination.place_name,
                    mode=TransportMode.TAXI,
                    duration_minutes=150 if long_first_day_route else 35,
                    distance_km=40 if long_first_day_route else 8,
                )
            )

    output = DataCollectorOutput(mcp_results=McpResults(weather=weather, attractions=attractions, routes=routes))
    return {**state, "mcp_results": output.mcp_results}


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

    for day in plan.days:
        day_weather = weather_by_date.get((day.city, day.date))
        for visit in day.visits:
            attraction = attraction_by_name.get(visit.place_name)
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

    for route in mcp_results.routes:
        if route.duration_minutes > 90:
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.ROUTE_TOO_LONG,
                    severity=Severity.HIGH,
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
    next_plan = current_plan.model_copy(deep=True)
    issue_types = {issue.issue_type for issue in agent_input.issues}
    bad_weather_locations = {
        location
        for issue in agent_input.issues
        if issue.issue_type == IssueType.BAD_WEATHER
        for location in issue.locations
    }
    closed_locations = {
        location
        for issue in agent_input.issues
        if issue.issue_type == IssueType.ATTRACTION_CLOSED
        for location in issue.locations
    }

    for day in next_plan.days:
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
            if visit.place_name in closed_locations:
                visit.place_name = f"{day.city} Tea House"
                visit.category = PlaceCategory.FOOD
                visit.notes = "Replanned from a closed attraction."
            if IssueType.ROUTE_TOO_LONG in issue_types and day.day == 1 and visit.transport_to_next:
                visit.transport_to_next.mode = TransportMode.TRANSIT
        day.daily_notes = "This day was adjusted by the mock replanner." if day.day == 1 else day.daily_notes

    planned_places = {visit.place_name for day in next_plan.days for visit in day.visits}
    for required_place in request.must_visit:
        if required_place not in planned_places:
            target_day = next((day for day in next_plan.days if day.day > 1), next_plan.days[0])
            if target_day.visits:
                target_day.visits[0].place_name = required_place
                target_day.visits[0].category = PlaceCategory.OUTDOOR
                target_day.visits[0].notes = "Moved here to avoid mock heavy rain on the first day."

    next_plan.assumptions = [*next_plan.assumptions, "One mock replanning pass was applied."]
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
