from __future__ import annotations

from typing import TypedDict

from backend.schemas.trip import FinalPlan, McpQueryPlan, McpResults, TripPlan, TripRequest, ValidationIssue


class TripState(TypedDict, total=False):
    raw_user_input: dict
    user_request: TripRequest
    pending_mcp_queries: McpQueryPlan
    current_plan: TripPlan
    plan_versions: list[TripPlan]
    mcp_results: McpResults
    issues: list[ValidationIssue]
    iteration: int
    max_iterations: int
    final_plan: FinalPlan
