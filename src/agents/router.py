# src/agents/router.py

from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm
from src.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from src.agents.state import SurgeryRAGState
from src.config import SUPPORTED_QUERY_TYPES


# =========================================================
# 1. STRUCTURED OUTPUT
# =========================================================

QueryType = Literal[
    "case_lookup",
    "instrument_lookup",
    "timing_duration",
    "pedal_activity",
    "comparison",
    "aggregation",
    "unsupported",
]

Operation = Literal[
    "lookup",
    "average",
    "minimum",
    "maximum",
    "count",
    "sum",
    "compare",
    "none",
]
SortBy = Literal[
    "surgery_started",
    "surgery_stopped",
    "none",
]
SortOrder = Literal[
    "asc",
    "desc",
    "none",
]

class RouterOutput(BaseModel):
    query_type: QueryType = Field(description="Classification of the user query.")
    operation: Operation = Field(default="none", description="Requested deterministic operation.")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters explicitly present in the question.")
    needs_aggregation: bool = Field(default=False, description="True when aggregation across multiple cases is required.")
    supported: bool = Field(default=True, description="False when the query cannot be answered from surgery logs.")
    limit: int | None = Field(default=None,description="Number of cases requested, such as last 2 or first 5.")
    sort_by: SortBy = Field(default="none",description="Field used to order cases.")
    sort_order: SortOrder = Field(default="none",description="asc for earliest/first, desc for latest/last.")


# =========================================================
# 2. ALLOWED FILTERS
# =========================================================

ALLOWED_FILTERS = {
    "case_id",
    "source_file",
    "surgery_type",
    "surgeon_name",
    "chunk_type",
}


# =========================================================
# 3. SHARED LLM
# =========================================================

llm = get_llm()
router_llm = llm.with_structured_output(RouterOutput)


# =========================================================
# 4. CLEAN FILTERS
# =========================================================

def clean_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    if not filters:
        return {}

    cleaned = {}

    for key, value in filters.items():
        if key not in ALLOWED_FILTERS or value is None:
            continue

        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue

        if isinstance(value, list):
            value = [item for item in value if item is not None and str(item).strip()]
            if not value:
                continue

        cleaned[key] = value

    return cleaned


# =========================================================
def normalize_routing(question: str, query_type: str, operation: str, filters: dict):
    q = question.lower()

    # Aggregation operations
    if any(word in q for word in ["total", "sum", "combined"]):
        operation = "sum"

        # If no specific case_id is provided, total means across matching cases
        if "case_id" not in filters:
            query_type = "aggregation"

    elif any(word in q for word in ["average", "mean"]):
        operation = "average"
        query_type = "aggregation"

    elif any(word in q for word in ["maximum", "highest", "longest"]):
        operation = "maximum"
        query_type = "aggregation"

    elif any(word in q for word in ["minimum", "lowest", "shortest"]):
        operation = "minimum"
        query_type = "aggregation"

    elif any(phrase in q for phrase in ["how many", "number of", "count"]):
        operation = "count"
        query_type = "aggregation"

    elif "compare" in q or "difference between" in q:
        operation = "compare"
        query_type = "comparison"

    # Duration lookup
    elif any(word in q for word in ["duration", "start time", "stop time"]):
        operation = "lookup"
        query_type = "timing_duration"

    return query_type, operation
# # =========================================================
# 5. ROUTER AGENT
# =========================================================

def router_agent(state: SurgeryRAGState) -> dict:

    question = state.get("question", "").strip()

    if not question:
        return {
            "query_type": "unsupported",
            "operation": "none",
            "filters": {},
            "limit": None,
            "sort_by": "none",
            "sort_order": "none",
            "needs_aggregation": False,
            "supported": False,
            "search_query": "",
            "retry_count": 0,
            "workflow_status": "failed",
            "error": "Question is empty.",
        }

    user_prompt = ROUTER_USER_PROMPT.format(question=question)

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        result = router_llm.invoke(messages)

        filters = clean_filters(result.filters)

        query_type = result.query_type if result.query_type in SUPPORTED_QUERY_TYPES else "unsupported"
        supported = result.supported and query_type != "unsupported"

        operation = result.operation
        query_type, operation = normalize_routing(question=question,query_type=query_type,operation=operation,filters=filters,)

        needs_aggregation = (
            result.needs_aggregation
            or query_type == "aggregation"
            or operation in {"average", "minimum", "maximum", "count", "sum"}
        )

        return {
            "query_type": query_type,
            "operation": operation,
            "filters": filters,
            "limit": result.limit,
            "sort_by": result.sort_by,
            "sort_order": result.sort_order,
            "needs_aggregation": needs_aggregation,
            "supported": supported,
            "search_query": question,
            "retry_count": state.get("retry_count", 0),
            "workflow_status": "routed" if supported else "unsupported",
            "error": None,
        }

    except Exception as exc:
        return {
            "query_type": "unsupported",
            "operation": "none",
            "filters": {},
            "limit": None,
            "sort_by": "none",
            "sort_order": "none",
            "needs_aggregation": False,
            "supported": False,
            "search_query": question,
            "retry_count": state.get("retry_count", 0),
            "workflow_status": "failed",
            "error": f"Router error: {str(exc)}",
        }


# =========================================================
# 6. LANGGRAPH ROUTING
# =========================================================

def route_after_router(state: SurgeryRAGState) -> str:
    if not state.get("supported", False):
        return "unsupported"

    return "retrieval"


# =========================================================
# 7. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    test_questions = [
        "Which instruments were used during Cholecystectomy?",
        "What was the surgery duration for the Cholecystectomy case?",
        "What is the average duration of Cholecystectomy surgeries?",
        "Which Cholecystectomy case had the longest duration?",
        "How many Cholecystectomy surgeries are recorded?",
        "Compare the surgery duration of Cholecystectomy and Appendectomy cases.",
        "How many times was the clutch pedal pressed?",
        "What is the weather in Chennai?",
    ]

    for question in test_questions:
        state = {"question": question, "retry_count": 0}
        result = router_agent(state)

        print("\n" + "=" * 70)
        print("Question:", question)
        print("Query type:", result.get("query_type"))
        print("Operation:", result.get("operation"))
        print("Filters:", result.get("filters"))
        print("Aggregation:", result.get("needs_aggregation"))
        print("Supported:", result.get("supported"))
        print("Status:", result.get("workflow_status"))
        print("Error:", result.get("error"))