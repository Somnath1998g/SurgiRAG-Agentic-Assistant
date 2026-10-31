from statistics import mean

from src.agents.state import SurgeryRAGState
from src.tools.analytics import (
    get_case_durations,
    select_cases,
    minutes_to_duration,
)


def analytics_agent(state: SurgeryRAGState) -> dict:

    query_type = state.get("query_type")
    operation = state.get("operation", "none")
    chunks = state.get("retrieved_chunks", [])

    limit = state.get("limit")
    sort_by = state.get("sort_by", "none")
    sort_order = state.get("sort_order", "none")

    # -----------------------------------------------------
    # Only run duration analytics when relevant
    # -----------------------------------------------------

    if query_type not in {"timing_duration", "aggregation"}:
        return {"analytics_result": {}}

    # -----------------------------------------------------
    # Extract unique cases
    # -----------------------------------------------------

    cases = get_case_durations(chunks)

    # -----------------------------------------------------
    # Apply first/last/latest N selection
    # -----------------------------------------------------

    cases = select_cases(
        cases,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    # -----------------------------------------------------
    # No usable cases
    # -----------------------------------------------------

    if not cases:
        return {
            "analytics_result": {
                "operation": operation,
                "count": 0,
                "cases": [],
            }
        }

    # -----------------------------------------------------
    # Collect durations AFTER case selection
    # -----------------------------------------------------

    durations = [
        case["duration_minutes"]
        for case in cases
        if case.get("duration_minutes") is not None
    ]

    if not durations:
        return {
            "analytics_result": {
                "operation": operation,
                "count": 0,
                "cases": cases,
            }
        }

    # -----------------------------------------------------
    # SUM
    # -----------------------------------------------------

    if operation == "sum":
        total_minutes = sum(durations)

        result = {
            "operation": "sum",
            "count": len(cases),
            "total_minutes": round(total_minutes, 2),
            "total_duration": minutes_to_duration(total_minutes),
            "cases": cases,
        }

    # -----------------------------------------------------
    # AVERAGE
    # -----------------------------------------------------

    elif operation == "average":
        average_minutes = mean(durations)

        result = {
            "operation": "average",
            "count": len(cases),
            "average_minutes": round(average_minutes, 2),
            "average_duration": minutes_to_duration(average_minutes),
            "cases": cases,
        }

    # -----------------------------------------------------
    # MAXIMUM
    # -----------------------------------------------------

    elif operation == "maximum":
        max_case = max(
            cases,
            key=lambda x: x["duration_minutes"],
        )

        result = {
            "operation": "maximum",
            "count": len(cases),
            "maximum_duration": max_case["duration"],
            "maximum_case": max_case,
            "cases": cases,
        }

    # -----------------------------------------------------
    # MINIMUM
    # -----------------------------------------------------

    elif operation == "minimum":
        min_case = min(
            cases,
            key=lambda x: x["duration_minutes"],
        )

        result = {
            "operation": "minimum",
            "count": len(cases),
            "minimum_duration": min_case["duration"],
            "minimum_case": min_case,
            "cases": cases,
        }

    # -----------------------------------------------------
    # COUNT
    # -----------------------------------------------------

    elif operation == "count":
        result = {
            "operation": "count",
            "count": len(cases),
            "cases": cases,
        }

    # -----------------------------------------------------
    # NORMAL LOOKUP
    # -----------------------------------------------------

    else:
        result = {
            "operation": "lookup",
            "count": len(cases),
            "cases": cases,
        }

    return {
        "analytics_result": result
    }