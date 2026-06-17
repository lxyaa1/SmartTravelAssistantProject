from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agents.nodes import (
    collect_mcp_data_node,
    final_writer_node,
    initial_plan_node,
    plan_check_query_planner_node,
    parse_request_node,
    preplan_query_planner_node,
    replan_node,
    validate_plan_node,
)
from backend.graph.routing import should_replan
from backend.graph.state import TripState


def build_workflow():
    graph = StateGraph(TripState)

    graph.add_node("parse_request", parse_request_node)
    graph.add_node("preplan_query_planner", preplan_query_planner_node)
    graph.add_node("collect_preplan_mcp_data", collect_mcp_data_node)
    graph.add_node("initial_plan", initial_plan_node)
    graph.add_node("plan_check_query_planner", plan_check_query_planner_node)
    graph.add_node("collect_plan_mcp_data", collect_mcp_data_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("replan", replan_node)
    graph.add_node("final_writer", final_writer_node)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "preplan_query_planner")
    graph.add_edge("preplan_query_planner", "collect_preplan_mcp_data")
    graph.add_edge("collect_preplan_mcp_data", "initial_plan")
    graph.add_edge("initial_plan", "plan_check_query_planner")
    graph.add_edge("plan_check_query_planner", "collect_plan_mcp_data")
    graph.add_edge("collect_plan_mcp_data", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        should_replan,
        {
            "replan": "replan",
            "final_writer": "final_writer",
        },
    )
    graph.add_edge("replan", "plan_check_query_planner")
    graph.add_edge("final_writer", END)

    return graph.compile()
