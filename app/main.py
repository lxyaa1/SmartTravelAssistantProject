from __future__ import annotations

from backend.graph.workflow import build_workflow


def main() -> None:
    workflow = build_workflow()
    result = workflow.invoke(
        {
            "raw_user_input": {
                "origin": "Shanghai",
                "destination": "Hangzhou",
                "start_date": "2026-07-01",
                "end_date": "2026-07-03",
                "travelers": {
                    "adults": 2,
                    "children": 1,
                    "children_need_bed": 0,
                    "children_ages": [6],
                },
                "budget_level": "medium",
                "preferences": ["culture", "food", "relaxed", "family friendly"],
                "must_visit": ["West Lake"],
                "avoid": ["overly packed schedule", "late night activities"],
            },
            "max_iterations": 3,
        }
    )

    final_plan = result["final_plan"]
    print(final_plan.content)
    if final_plan.unresolved_issues:
        print("\nUnresolved issues:")
        for issue in final_plan.unresolved_issues:
            print(f"- {issue.severity}: {issue.reason}")


if __name__ == "__main__":
    main()
