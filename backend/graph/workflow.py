from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agents.nodes import (
    collect_mcp_data_node,
    final_writer_node,
    initial_plan_node,
    parse_request_node,
    replan_node,
    validate_plan_node,
)
from backend.graph.routing import should_replan
from backend.graph.state import TripState


def build_workflow():
    graph = StateGraph(TripState)

    graph.add_node("parse_request", parse_request_node)
    graph.add_node("initial_plan", initial_plan_node)
    graph.add_node("collect_mcp_data", collect_mcp_data_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("replan", replan_node)
    graph.add_node("final_writer", final_writer_node)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "initial_plan")
    graph.add_edge("initial_plan", "collect_mcp_data")
    graph.add_edge("collect_mcp_data", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        should_replan,
        {
            "replan": "replan",
            "final_writer": "final_writer",
        },
    )
    graph.add_edge("replan", "collect_mcp_data")
    graph.add_edge("final_writer", END)

    return graph.compile()
