from __future__ import annotations

from backend.graph.state import TripState
from backend.schemas.trip import Severity


def should_replan(state: TripState) -> str:
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "final_writer"

    serious_issues = [
        issue
        for issue in state.get("issues", [])
        if issue.severity in {Severity.HIGH, Severity.CRITICAL}
    ]
    return "replan" if serious_issues else "final_writer"
