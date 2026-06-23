from __future__ import annotations

from datetime import date, time

import backend.agents.nodes as nodes
from backend.agents.nodes import (
    city_route_planner_node,
    collect_mcp_data_node,
    plan_check_query_planner_node,
    validate_plan_node,
)
from backend.graph.workflow import build_workflow
from backend.schemas.trip import (
    FinalPlan,
    IssueType,
    LodgingResult,
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
    WeatherResult,
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
    assert all(issue.issue_type != IssueType.TIME_CONFLICT for issue in result["issues"])
    assert any(
        item.stay and item.stay.place_name == "West Lake" and "weather" in item.stay.notes.lower()
        for day in plan.days
        for item in day.timeline
    )


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
    city_scopes = {
        (query.args["origin"], query.args["destination"]): (query.args["origin_city"], query.args["destination_city"])
        for query in route_queries
    }

    assert ("Beijing", "Shanxi") in route_pairs
    assert ("Taiyuan Hotel", "Wutai Mountain") in route_pairs
    assert ("Wutai Mountain", "Taiyuan Hotel") in route_pairs
    assert city_scopes[("Beijing", "Shanxi")] == ("Beijing", "Taiyuan")
    assert city_scopes[("Taiyuan Hotel", "Wutai Mountain")] == ("Taiyuan", "Xinzhou")
    assert city_scopes[("Wutai Mountain", "Taiyuan Hotel")] == ("Xinzhou", "Taiyuan")


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
    assert result["mcp_cache_stats"]["last_skipped_existing_results"] >= 1


def test_validator_updates_move_duration_from_mcp_route_results_and_shifts_timeline() -> None:
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
    visit = _first_visit(plan.days[0])
    assert visit.start_time == time(13, 30)
    assert result["issues"]
    assert any(issue.reason == "Local move takes 180 minutes." for issue in result["issues"])
    assert any(issue.issue_type.value == "missing_return_transfer" for issue in result["issues"])
    assert all(issue.issue_type != IssueType.TIME_CONFLICT for issue in result["issues"])


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


def test_best_lodging_prefers_located_candidate_over_missing_coordinates() -> None:
    results = McpResults(
        lodging=[
            LodgingResult(
                name="No Coordinate Hotel",
                city="Xinzhou",
                anchor_place="Wutai Mountain",
                distance_to_anchor_km=0,
            ),
            LodgingResult(
                name="Located Hotel",
                city="Xinzhou",
                anchor_place="Wutai Mountain",
                location="113.5,39.0",
                distance_to_anchor_km=2,
                duration_to_anchor_minutes=15,
            ),
        ]
    )

    selected = nodes._best_lodging_for_anchor(results, "Xinzhou", "Wutai Mountain")

    assert selected is not None
    assert selected.name == "Located Hotel"


def test_best_lodging_rejects_candidates_without_trustworthy_location_data() -> None:
    results = McpResults(
        lodging=[
            LodgingResult(
                name="No Coordinate Hotel",
                city="Guilin",
                anchor_place="Elephant Trunk Hill",
                distance_to_anchor_km=0,
                duration_to_anchor_minutes=0,
            ),
            LodgingResult(
                name="Far Located Inn",
                city="Guilin",
                anchor_place="Elephant Trunk Hill",
                location="110.1,25.1",
                distance_to_anchor_km=188,
                duration_to_anchor_minutes=189,
            ),
        ]
    )

    assert nodes._best_lodging_for_anchor(results, "Guilin", "Elephant Trunk Hill") is None


def test_timeline_dedupe_drops_items_that_cannot_fit_before_midnight() -> None:
    first = TimelineItem(
        sequence=1,
        item_type=TimelineItemType.STAY,
        start_time=time(23, 0),
        end_time=time(23, 59),
        city="Chengdu",
        stay=StayDetail(
            place_name="Chengdu",
            city="Chengdu",
            purpose=StayPurpose.REST,
            duration_minutes=59,
        ),
    )
    overlapping = TimelineItem(
        sequence=2,
        item_type=TimelineItemType.MOVE,
        start_time=time(23, 58),
        end_time=time(23, 59),
        city="Chengdu",
        move=MoveDetail(
            origin="Chengdu",
            destination="Hotel",
            mode=TransportMode.TAXI,
            purpose=MovePurpose.LOCAL,
            duration_minutes=1,
        ),
    )

    deduped = nodes._dedupe_and_sort_timeline([first, overlapping])

    assert deduped == [first]


def test_unverified_lodging_moves_are_removed_from_timeline() -> None:
    day = PlanDay(
        day=1,
        date=date(2026, 9, 5),
        city="Guilin",
        accommodation_area="Guilin lodging unresolved near Elephant Trunk Hill",
        timeline=[
            TimelineItem(
                sequence=1,
                item_type=TimelineItemType.MOVE,
                start_time=time(9, 0),
                end_time=time(9, 35),
                city="Guilin",
                move=MoveDetail(
                    origin="Guilin lodging unresolved near Elephant Trunk Hill",
                    destination="Elephant Trunk Hill",
                    mode=TransportMode.TAXI,
                    purpose=MovePurpose.LOCAL,
                    duration_minutes=35,
                ),
            ),
            TimelineItem(
                sequence=2,
                item_type=TimelineItemType.STAY,
                start_time=time(9, 35),
                end_time=time(11, 35),
                city="Guilin",
                stay=StayDetail(
                    place_name="Elephant Trunk Hill",
                    city="Guilin",
                    purpose=StayPurpose.VISIT,
                    category=PlaceCategory.OUTDOOR,
                    duration_minutes=120,
                ),
            ),
        ],
    )

    nodes._remove_unverified_lodging_moves(day)

    assert all(not item.move for item in day.timeline)


def test_bad_weather_repair_replaces_optional_outdoor_visit() -> None:
    plan = _plan_with_daily_moves()
    day = plan.days[0]
    day.timeline.append(
        TimelineItem(
            sequence=5,
            item_type=TimelineItemType.STAY,
            start_time=time(14, 0),
            end_time=time(15, 0),
            city="Shanxi",
            stay=StayDetail(
                place_name="Optional Park",
                city="Shanxi",
                purpose=StayPurpose.VISIT,
                category=PlaceCategory.OUTDOOR,
                duration_minutes=60,
            ),
        )
    )
    request = TripRequest(
        origin="Beijing",
        destination="Shanxi",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        must_visit=["Wutai Mountain"],
    )

    nodes._move_rainy_outdoor_visits(
        plan,
        request,
        McpResults(weather=[WeatherResult(city="Shanxi", date=date(2026, 7, 1), condition="heavy rain")]),
    )

    visit_names = _visit_names(plan)
    assert "Wutai Mountain" in visit_names
    assert "Optional Park" not in visit_names
    assert any(
        item.stay
        and item.stay.purpose == StayPurpose.REST
        and "no verified indoor replacement" in item.stay.notes
        for item in day.timeline
    )


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


def test_validator_flags_suspicious_cross_city_short_route() -> None:
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
                        origin_city="Taiyuan",
                        destination_city="Xinzhou",
                        mode=TransportMode.TAXI,
                        duration_minutes=15,
                        distance_km=8,
                    )
                ]
            ),
        }
    )

    assert any(
        issue.issue_type.value == "infeasible_plan"
        and "Cross-city route result is suspicious" in issue.reason
        for issue in result["issues"]
    )


def test_validator_blocks_plan_when_move_route_data_is_missing() -> None:
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
            "mcp_results": McpResults(),
        }
    )

    assert plan.quality_gate.can_finalize is False
    assert any(issue.issue_type == IssueType.MISSING_MCP_DATA for issue in result["issues"])
    assert any("unverified" in issue.reason for issue in result["issues"])


def test_validator_flags_destination_activity_after_return() -> None:
    plan = _plan_with_daily_moves()
    last_day = plan.days[0]
    last_day.timeline.append(
        TimelineItem(
            sequence=5,
            item_type=TimelineItemType.MOVE,
            start_time=time(17, 0),
            end_time=time(21, 0),
            city="Shanxi",
            move=MoveDetail(
                origin="Taiyuan",
                destination="Beijing",
                origin_city="Taiyuan",
                destination_city="Beijing",
                mode=TransportMode.TRAIN,
                purpose=MovePurpose.RETURN,
                duration_minutes=240,
                distance_km=490,
            ),
        )
    )
    last_day.timeline.append(
        TimelineItem(
            sequence=6,
            item_type=TimelineItemType.STAY,
            start_time=time(21, 30),
            end_time=time(22, 30),
            city="Xinzhou",
            stay=StayDetail(
                place_name="Wutai Mountain",
                city="Xinzhou",
                purpose=StayPurpose.VISIT,
                category=PlaceCategory.OUTDOOR,
                duration_minutes=60,
            ),
        )
    )
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
            "mcp_results": McpResults(),
        }
    )

    assert any("after the return move" in issue.reason for issue in result["issues"])


def test_validator_flags_first_item_not_starting_at_origin() -> None:
    plan = _plan_with_daily_moves()
    plan.days[0].timeline.insert(
        0,
        TimelineItem(
            sequence=1,
            item_type=TimelineItemType.STAY,
            start_time=time(0, 0),
            end_time=time(7, 30),
            city="Taiyuan",
            stay=StayDetail(
                place_name="Taiyuan Hotel",
                city="Taiyuan",
                purpose=StayPurpose.SLEEP,
                category=PlaceCategory.HOTEL_AREA,
                duration_minutes=450,
            ),
        ),
    )
    for index, item in enumerate(plan.days[0].timeline, start=1):
        item.sequence = index
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
            "mcp_results": McpResults(),
        }
    )

    assert any(
        issue.severity.value == "critical"
        and "first chronological timeline item does not start" in issue.reason
        for issue in result["issues"]
    )


def test_validator_flags_adjacent_timeline_location_break() -> None:
    plan = _plan_with_daily_moves()
    plan.days[0].timeline[1].move.origin = "Datong Hotel"
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
            "mcp_results": McpResults(),
        }
    )

    assert _move_between(plan.days[0], "Shanxi", "Datong Hotel") is not None
    assert not any("Adjacent timeline items do not connect" in issue.reason for issue in result["issues"])


def test_collect_mcp_data_does_not_fallback_to_mock_after_amap_failure(monkeypatch) -> None:
    request = TripRequest(
        origin="Beijing",
        destination="Shanxi",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
    )
    query_plan = McpQueryPlan(
        queries=[
            McpQuery(
                tool_name=McpToolName.GET_ROUTE_TIME,
                args={
                    "origin": "Taiyuan",
                    "destination": "Wutai Mountain",
                    "origin_city": "Taiyuan",
                    "destination_city": "Xinzhou",
                    "mode": "transit",
                },
                purpose="test failed amap batch",
                stage=McpQueryStage.PLAN_CHECK,
            )
        ]
    )

    def fail_amap(*args, **kwargs):
        raise RuntimeError("CUQPS_HAS_EXCEEDED_THE_LIMIT")

    monkeypatch.setattr(nodes, "execute_amap_mcp_query_plan", fail_amap)
    result = collect_mcp_data_node(
        {
            "user_request": request,
            "mcp_backend": "amap",
            "pending_mcp_queries": query_plan,
            "mcp_results": McpResults(),
        }
    )

    assert result["mcp_results"].routes == []
    assert result["mcp_cache_stats"]["entries"] == 0
    assert any("no mock data was mixed in" in error for error in result["mcp_errors"])


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
                origin_city="Beijing",
                destination_city="Taiyuan",
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
                            origin_city="Beijing",
                            destination_city="Taiyuan",
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
                            origin_city="Taiyuan",
                            destination_city="Xinzhou",
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
                            origin_city="Xinzhou",
                            destination_city="Taiyuan",
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
