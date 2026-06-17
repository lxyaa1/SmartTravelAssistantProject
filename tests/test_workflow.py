from __future__ import annotations

from backend.graph.workflow import build_workflow
from backend.schemas.trip import FinalPlan, McpQueryPlan, McpResults, TripPlan, ValidationIssue


def test_workflow_replans_and_returns_final_plan() -> None:
    workflow = build_workflow()
    result = workflow.invoke(
        {
            "use_llm": False,
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
