from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agents.nodes import (
    city_route_planner_node,
    collect_mcp_data_node,
    draft_day_schedule_node,
    final_writer_node,
    plan_check_query_planner_node,
    parse_request_node,
    preplan_query_planner_node,
    repair_strategy_planner_node,
    replan_node,
    validate_plan_node,
)
from backend.graph.routing import route_after_repair_strategy
from backend.graph.state import TripState


def build_workflow():
    graph = StateGraph(TripState)

    graph.add_node("parse_request", parse_request_node)
    graph.add_node("city_route_planner", city_route_planner_node)
    graph.add_node("preplan_query_planner", preplan_query_planner_node)
    graph.add_node("collect_preplan_mcp_data", collect_mcp_data_node)
    graph.add_node("draft_day_schedule", draft_day_schedule_node)
    graph.add_node("plan_check_query_planner", plan_check_query_planner_node)
    graph.add_node("collect_plan_mcp_data", collect_mcp_data_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("repair_strategy_planner", repair_strategy_planner_node)
    graph.add_node("replan", replan_node)
    graph.add_node("final_writer", final_writer_node)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "city_route_planner")
    graph.add_edge("city_route_planner", "preplan_query_planner")
    graph.add_edge("preplan_query_planner", "collect_preplan_mcp_data")
    graph.add_edge("collect_preplan_mcp_data", "draft_day_schedule")
    graph.add_edge("draft_day_schedule", "plan_check_query_planner")
    graph.add_edge("plan_check_query_planner", "collect_plan_mcp_data")
    graph.add_edge("collect_plan_mcp_data", "validate_plan")
    graph.add_edge("validate_plan", "repair_strategy_planner")
    graph.add_conditional_edges(
        "repair_strategy_planner",
        route_after_repair_strategy,
        {
            "replan": "replan",
            "final_writer": "final_writer",
        },
    )
    graph.add_edge("replan", "plan_check_query_planner")
    graph.add_edge("final_writer", END)

    return graph.compile()
