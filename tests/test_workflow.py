from __future__ import annotations

from datetime import date, time

from backend.agents.nodes import plan_check_query_planner_node, validate_plan_node
from backend.graph.workflow import build_workflow
from backend.schemas.trip import (
    FinalPlan,
    McpQueryPlan,
    McpResults,
    McpToolName,
    PlaceCategory,
    PlanDay,
    RouteResult,
    TransferLeg,
    TransportMode,
    TripPlan,
    TripRequest,
    ValidationIssue,
    VisitSlot,
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

    assert result["final_plan"].content
    assert result["iteration"] >= 1
    assert result["plan_versions"]
    assert isinstance(result["current_plan"], TripPlan)
    assert all(isinstance(plan, TripPlan) for plan in result["plan_versions"])
    assert isinstance(result["mcp_results"], McpResults)
    assert result["mcp_results"].weather
    assert result["mcp_results"].attractions
    assert result["mcp_results"].routes
    assert result["mcp_results"].accommodation_areas
    assert all(isinstance(issue, ValidationIssue) for issue in result["issues"])
    assert isinstance(result["final_plan"], FinalPlan)
    assert isinstance(result["pending_mcp_queries"], McpQueryPlan)
    assert result["pending_mcp_queries"].queries == []
    assert result["current_plan"].days[0].visits[0].place_name == "Hangzhou Art Gallery"


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


def test_plan_check_queries_intercity_and_accommodation_transfers() -> None:
    plan = _plan_with_daily_transfers()
    plan.days[0].visits[0].transport_to_next = TransferLeg(
        origin="Wutai Mountain",
        destination="Taiyuan Hotel",
        mode=TransportMode.TAXI,
        estimated_duration_minutes=35,
        estimated_distance_km=0,
        estimated_cost=30,
    )

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


def test_validator_updates_transfer_duration_from_mcp_route_results() -> None:
    plan = _plan_with_daily_transfers()
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

    assert plan.days[0].start_transfer_to_first is not None
    assert plan.days[0].start_transfer_to_first.estimated_duration_minutes == 180
    assert result["issues"]
    assert result["issues"][0].reason == "start transfer takes 180 minutes."


def test_validator_accepts_normalized_must_visit_names() -> None:
    plan = _plan_with_daily_transfers()
    plan.days[0].visits[0].place_name = "云冈石窟"
    request = TripRequest(
        origin="Beijing",
        destination="Shanxi",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        travelers=3,
        must_visit=["云岗石窟"],
    )

    result = validate_plan_node(
        {
            "user_request": request,
            "current_plan": plan,
            "mcp_results": McpResults(),
        }
    )

    assert all(issue.issue_type.value != "missing_must_visit" for issue in result["issues"])


def _plan_with_daily_transfers() -> TripPlan:
    visit = VisitSlot(
        sequence=1,
        place_name="Wutai Mountain",
        city="Shanxi",
        category=PlaceCategory.OUTDOOR,
        start_time=time(9, 0),
        end_time=time(11, 0),
        visit_duration_minutes=120,
    )
    return TripPlan(
        title="Beijing to Shanxi",
        origin="Beijing",
        destination="Shanxi",
        days=[
            PlanDay(
                day=1,
                date=date(2026, 7, 1),
                city="Shanxi",
                accommodation_area="Taiyuan Hotel",
                visits=[visit],
                arrival_transfer=TransferLeg(
                    origin="Beijing",
                    destination="Shanxi",
                    mode=TransportMode.TRAIN,
                    estimated_duration_minutes=120,
                    estimated_distance_km=0,
                    estimated_cost=150,
                ),
                start_transfer_to_first=TransferLeg(
                    origin="Taiyuan Hotel",
                    destination="Wutai Mountain",
                    mode=TransportMode.TAXI,
                    estimated_duration_minutes=35,
                    estimated_distance_km=0,
                    estimated_cost=30,
                ),
                return_transfer_to_accommodation=TransferLeg(
                    origin="Wutai Mountain",
                    destination="Taiyuan Hotel",
                    mode=TransportMode.TAXI,
                    estimated_duration_minutes=35,
                    estimated_distance_km=0,
                    estimated_cost=30,
                ),
            )
        ],
    )
