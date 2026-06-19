from __future__ import annotations

from datetime import date, time

from backend.agents.nodes import (
    city_route_planner_node,
    collect_mcp_data_node,
    plan_check_query_planner_node,
    validate_plan_node,
)
from backend.graph.workflow import build_workflow
from backend.schemas.trip import (
    FinalPlan,
    McpQuery,
    McpQueryPlan,
    McpQueryStage,
    McpResults,
    McpToolName,
    MoveDetail,
    MovePurpose,
    PlaceCategory,
    PlanDay,
    RouteResult,
    SegmentType,
    StayDetail,
    StayPurpose,
    TimelineItem,
    TimelineItemType,
    TransportMode,
    TripPlan,
    TripRequest,
    TripSegment,
    ValidationIssue,
)


def test_workflow_replans_and_returns_final_plan() -> None:
    workflow = build_workflow()
    result = workflow.invoke(
        {
            "use_llm": False,
            "mcp_backend": "mock",
            "raw_user_input": {
                "origin": "Shanghai",
                "destination": "Hangzhou",
                "start_date": "2026-07-01",
                "end_date": "2026-07-03",
                "travelers": 2,
                "budget_level": "medium",
                "preferences": ["culture", "food"],
                "must_visit": ["West Lake"],
            },
            "max_iterations": 3,
        }
    )

    plan = result["current_plan"]
    assert result["final_plan"].content
    assert result["iteration"] >= 1
    assert result["plan_versions"]
    assert isinstance(plan, TripPlan)
    assert result["city_route_plan"].segments
    assert plan.route_segments
    assert plan.accommodations
    assert all(day.timeline for day in plan.days)
    assert all(isinstance(plan_version, TripPlan) for plan_version in result["plan_versions"])
    assert isinstance(result["mcp_results"], McpResults)
    assert result["mcp_results"].weather
    assert result["mcp_results"].attractions
    assert result["mcp_results"].routes
    assert result["mcp_results"].accommodation_areas
    assert result["mcp_results"].lodging
    assert all(isinstance(issue, ValidationIssue) for issue in result["issues"])
    assert isinstance(result["final_plan"], FinalPlan)
    assert isinstance(result["pending_mcp_queries"], McpQueryPlan)
    assert result["pending_mcp_queries"].queries == []
    assert "West Lake" in _visit_names(plan)
    assert _first_visit_name(plan.days[0]) == "Hangzhou Art Gallery"


def test_workflow_stops_at_max_iterations_and_keeps_issues() -> None:
    workflow = build_workflow()
    result = workflow.invoke(
        {
            "use_llm": False,
            "mcp_backend": "mock",
            "raw_user_input": {
                "origin": "Shanghai",
                "destination": "Hangzhou",
                "start_date": "2026-07-01",
                "end_date": "2026-07-03",
                "travelers": 2,
                "budget_level": "medium",
                "preferences": ["culture", "food"],
                "must_visit": ["West Lake"],
            },
            "max_iterations": 0,
        }
    )

    assert result["iteration"] == 0
    assert result["issues"]
    assert result["final_plan"].unresolved_issues == result["issues"]


def test_plan_check_queries_route_segments_and_timeline_moves() -> None:
    plan = _plan_with_daily_moves()

    result = plan_check_query_planner_node({"current_plan": plan})
    route_queries = [
        query
        for query in result["pending_mcp_queries"].queries
        if query.tool_name == McpToolName.GET_ROUTE_TIME
    ]
    route_pairs = {(query.args["origin"], query.args["destination"]) for query in route_queries}

    assert ("Beijing", "Shanxi") in route_pairs
    assert ("Taiyuan Hotel", "Wutai Mountain") in route_pairs
    assert ("Wutai Mountain", "Taiyuan Hotel") in route_pairs


def test_plan_check_skips_route_queries_already_in_mcp_results() -> None:
    plan = _plan_with_daily_moves()
    result = plan_check_query_planner_node(
        {
            "current_plan": plan,
            "mcp_results": McpResults(
                routes=[
                    RouteResult(
                        origin="Taiyuan Hotel",
                        destination="Wutai Mountain",
                        mode=TransportMode.TAXI,
                        duration_minutes=180,
                        distance_km=150,
                    )
                ]
            ),
        }
    )

    route_queries = [
        query
        for query in result["pending_mcp_queries"].queries
        if query.tool_name == McpToolName.GET_ROUTE_TIME
    ]
    route_pairs = {(query.args["origin"], query.args["destination"]) for query in route_queries}

    assert ("Taiyuan Hotel", "Wutai Mountain") not in route_pairs


def test_validator_updates_move_duration_from_mcp_route_results() -> None:
    plan = _plan_with_daily_moves()
    request = TripRequest(
        origin="Beijing",
        destination="Shanxi",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        travelers=3,
        must_visit=["Wutai Mountain"],
    )

    result = validate_plan_node(
        {
            "user_request": request,
            "current_plan": plan,
            "mcp_results": McpResults(
                routes=[
                    RouteResult(
                        origin="Taiyuan Hotel",
                        destination="Wutai Mountain",
                        mode=TransportMode.TAXI,
                        duration_minutes=180,
                        distance_km=150,
                    )
                ]
            ),
        }
    )

    move = _move_between(plan.days[0], "Taiyuan Hotel", "Wutai Mountain")
    assert move is not None
    assert move.move is not None
    assert move.move.duration_minutes == 180
    assert result["issues"]
    assert any(issue.reason == "Local move takes 180 minutes." for issue in result["issues"])
    assert any(issue.issue_type.value == "missing_return_transfer" for issue in result["issues"])


def test_collect_mcp_data_reuses_query_cache() -> None:
    request = TripRequest(
        origin="Shanghai",
        destination="Hangzhou",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
    )
    query_plan = McpQueryPlan(
        queries=[
            McpQuery(
                tool_name=McpToolName.GET_WEATHER,
                args={"city": "Hangzhou", "date": "2026-07-01"},
                purpose="test cache",
                stage=McpQueryStage.PREPLAN,
            )
        ]
    )

    first = collect_mcp_data_node(
        {
            "user_request": request,
            "mcp_backend": "mock",
            "pending_mcp_queries": query_plan,
            "mcp_results": McpResults(),
        }
    )
    second = collect_mcp_data_node(
        {
            "user_request": request,
            "mcp_backend": "mock",
            "pending_mcp_queries": query_plan,
            "mcp_results": McpResults(),
            "mcp_cache": first["mcp_cache"],
        }
    )

    assert first["mcp_cache_stats"]["misses"] == 1
    assert second["mcp_cache_stats"]["hits"] == 1
    assert second["mcp_results"].weather[0].city == "Hangzhou"


def test_city_route_planner_adds_concrete_stay_and_return_segment() -> None:
    request = TripRequest(
        origin="Beijing",
        destination="Shanxi",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        travelers=3,
        must_visit=["Wutai Mountain"],
    )

    result = city_route_planner_node({"user_request": request, "use_llm": False})
    route = result["city_route_plan"]

    assert route.stays[0].city == "Xinzhou"
    assert route.stays[0].lodging_anchor == "Wutai Mountain"
    assert any(segment.segment_type.value == "outbound" for segment in route.segments)
    assert any(segment.segment_type.value == "return" for segment in route.segments)


def test_validator_accepts_normalized_must_visit_names() -> None:
    plan = _plan_with_daily_moves()
    _first_visit(plan.days[0]).stay.place_name = "Yungang Grottoes"
    request = TripRequest(
        origin="Beijing",
        destination="Shanxi",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        travelers=3,
        must_visit=["Yungang Grotto"],
    )

    result = validate_plan_node(
        {
            "user_request": request,
            "current_plan": plan,
            "mcp_results": McpResults(),
        }
    )

    assert all(issue.issue_type.value != "missing_must_visit" for issue in result["issues"])


def _plan_with_daily_moves() -> TripPlan:
    return TripPlan(
        title="Beijing to Shanxi",
        origin="Beijing",
        destination="Shanxi",
        route_segments=[
            TripSegment(
                sequence=1,
                segment_type=SegmentType.OUTBOUND,
                origin="Beijing",
                destination="Shanxi",
                mode=TransportMode.TRAIN,
                departure_date=date(2026, 7, 1),
                departure_time=time(8, 0),
                arrival_date=date(2026, 7, 1),
                arrival_time=time(10, 0),
                estimated_duration_minutes=120,
            )
        ],
        days=[
            PlanDay(
                day=1,
                date=date(2026, 7, 1),
                city="Shanxi",
                accommodation_area="Taiyuan Hotel",
                timeline=[
                    TimelineItem(
                        sequence=1,
                        item_type=TimelineItemType.MOVE,
                        start_time=time(8, 0),
                        end_time=time(10, 0),
                        city="Shanxi",
                        move=MoveDetail(
                            origin="Beijing",
                            destination="Shanxi",
                            mode=TransportMode.TRAIN,
                            purpose=MovePurpose.OUTBOUND,
                            duration_minutes=120,
                        ),
                    ),
                    TimelineItem(
                        sequence=2,
                        item_type=TimelineItemType.MOVE,
                        start_time=time(10, 30),
                        end_time=time(11, 5),
                        city="Shanxi",
                        move=MoveDetail(
                            origin="Taiyuan Hotel",
                            destination="Wutai Mountain",
                            mode=TransportMode.TAXI,
                            purpose=MovePurpose.LOCAL,
                            duration_minutes=35,
                        ),
                    ),
                    TimelineItem(
                        sequence=3,
                        item_type=TimelineItemType.STAY,
                        start_time=time(11, 5),
                        end_time=time(13, 5),
                        city="Shanxi",
                        stay=StayDetail(
                            place_name="Wutai Mountain",
                            city="Shanxi",
                            purpose=StayPurpose.VISIT,
                            category=PlaceCategory.OUTDOOR,
                            duration_minutes=120,
                        ),
                    ),
                    TimelineItem(
                        sequence=4,
                        item_type=TimelineItemType.MOVE,
                        start_time=time(13, 5),
                        end_time=time(13, 40),
                        city="Shanxi",
                        move=MoveDetail(
                            origin="Wutai Mountain",
                            destination="Taiyuan Hotel",
                            mode=TransportMode.TAXI,
                            purpose=MovePurpose.LOCAL,
                            duration_minutes=35,
                        ),
                    ),
                ],
            )
        ],
    )


def _visit_names(plan: TripPlan) -> list[str]:
    return [
        item.stay.place_name
        for day in plan.days
        for item in day.timeline
        if item.stay and item.stay.purpose == StayPurpose.VISIT
    ]


def _first_visit_name(day: PlanDay) -> str:
    return _first_visit(day).stay.place_name


def _first_visit(day: PlanDay) -> TimelineItem:
    item = next(item for item in day.timeline if item.stay and item.stay.purpose == StayPurpose.VISIT)
    assert item.stay is not None
    return item


def _move_between(day: PlanDay, origin: str, destination: str) -> TimelineItem | None:
    return next(
        (
            item
            for item in day.timeline
            if item.move and item.move.origin == origin and item.move.destination == destination
        ),
        None,
    )
