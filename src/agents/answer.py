# src/agents/answer.py

import time
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm
from src.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT
from src.config import FALLBACK_RESPONSE
from src.agents.state import SurgeryRAGState


# =========================================================
# 1. LOAD SHARED LLM
# =========================================================

llm = get_llm()


# =========================================================
# 2. ANSWER AGENT
# =========================================================

def answer_agent(state: SurgeryRAGState) -> dict:

    question = state.get("question", "").strip()
    context = state.get("context", "").strip()
    analytics_result = state.get("analytics_result", {})
    operation = state.get("operation", "none")
    query_type = state.get("query_type", "")


    # =====================================================
    # 3. BASIC VALIDATION
    # =====================================================

    if not question:
        return {
            "answer": FALLBACK_RESPONSE,
            "workflow_status": "failed",
            "llm_latency_ms": None,
            "error": "Question is empty.",
        }

    if not context and not analytics_result:
        return {
            "answer": FALLBACK_RESPONSE,
            "workflow_status": "answered",
            "llm_latency_ms": None,
            "error": None,
        }


    # =====================================================
    # 4. DETERMINISTIC ANALYTICS ANSWERS
    # =====================================================

    if query_type == "aggregation" and analytics_result:

        if operation == "sum" and analytics_result.get("total_duration"):
            count = analytics_result.get("count", 0)
            total_minutes = analytics_result.get("total_minutes")
            total_duration = analytics_result.get("total_duration")

            return {
                "answer": f"The total recorded surgery duration across {count} matching cases is {total_duration} ({total_minutes:g} minutes).",
                "workflow_status": "answered",
                "llm_latency_ms": 0.0,
                "error": None,
            }

        if operation == "average" and analytics_result.get("average_duration"):
            count = analytics_result.get("count", 0)
            average_duration = analytics_result.get("average_duration")
            average_minutes = analytics_result.get("average_minutes")

            answer = f"The average recorded surgery duration across {count} matching cases is {average_duration}."

            if average_minutes is not None:
                answer += f" This is approximately {average_minutes:g} minutes."

            return {
                "answer": answer,
                "workflow_status": "answered",
                "llm_latency_ms": 0.0,
                "error": None,
            }

        if operation == "maximum" and analytics_result.get("maximum_duration"):
            maximum_duration = analytics_result.get("maximum_duration")
            maximum_case = analytics_result.get("maximum_case", {})
            case_id = maximum_case.get("case_id")

            answer = f"The maximum recorded surgery duration is {maximum_duration}."

            if case_id:
                answer += f" It belongs to case {case_id}."

            return {
                "answer": answer,
                "workflow_status": "answered",
                "llm_latency_ms": 0.0,
                "error": None,
            }

        if operation == "minimum" and analytics_result.get("minimum_duration"):
            minimum_duration = analytics_result.get("minimum_duration")
            minimum_case = analytics_result.get("minimum_case", {})
            case_id = minimum_case.get("case_id")

            answer = f"The minimum recorded surgery duration is {minimum_duration}."

            if case_id:
                answer += f" It belongs to case {case_id}."

            return {
                "answer": answer,
                "workflow_status": "answered",
                "llm_latency_ms": 0.0,
                "error": None,
            }

        if operation == "count" and analytics_result.get("count") is not None:
            count = analytics_result.get("count", 0)

            return {
                "answer": f"There are {count} matching cases.",
                "workflow_status": "answered",
                "llm_latency_ms": 0.0,
                "error": None,
            }


    # =====================================================
    # 5. BUILD LLM ANSWER PROMPT
    # =====================================================

    user_prompt = ANSWER_USER_PROMPT.format(
        question=question,
        context=context,
        analytics_result=analytics_result,
    )

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]


    # =====================================================
    # 6. GENERATE NORMAL RAG ANSWER
    # =====================================================

    try:
        start_time = time.perf_counter()
        response = llm.invoke(messages)
        llm_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        answer = getattr(response, "content", "")

        if not answer:
            answer = FALLBACK_RESPONSE

        answer = str(answer).strip()

        return {
            "answer": answer,
            "llm_latency_ms": llm_latency_ms,
            "workflow_status": "answered",
            "error": None,
        }

    except Exception as exc:
        return {
            "answer": FALLBACK_RESPONSE,
            "llm_latency_ms": None,
            "workflow_status": "failed",
            "error": f"Answer generation error: {str(exc)}",
        }


# =========================================================
# 7. SIMPLE TEST - NORMAL RETRIEVAL
# =========================================================

if __name__ == "__main__":

    test_state = {
        "question": "Which instruments were used during the Cholecystectomy case?",
        "query_type": "instrument_lookup",
        "operation": "lookup",
        "context": """
Retrieved Chunk 1

Case ID:
Cholecystectomy_2025-10-01_12-34-00

Surgery Type:
Cholecystectomy

Surgeon:
Dr MERAI

Chunk Type:
instrument_summary

Source File:
Cholecystectomy_2025-10-01_12-34-00.json

Content:
Primary left instrument: Cadiere Forceps.
Primary right instrument: Monopolar Curved Scissors.
        """,
        "analytics_result": {},
        "retry_count": 0,
    }

    result = answer_agent(test_state)

    print("\nGenerated Answer:", result.get("answer"))
    print("LLM latency:", result.get("llm_latency_ms"), "ms")
    print("Status:", result.get("workflow_status"))
    print("Error:", result.get("error"))


    # =====================================================
    # 8. TEST - SUM
    # =====================================================

    sum_test_state = {
        "question": "What was the total surgery duration for the Cholecystectomy cases?",
        "query_type": "aggregation",
        "operation": "sum",
        "context": "Retrieved Cholecystectomy cases with recorded durations.",
        "analytics_result": {
            "operation": "sum",
            "count": 5,
            "total_minutes": 362.0,
            "total_duration": "6 h 2 min",
        },
        "retry_count": 0,
    }

    sum_result = answer_agent(sum_test_state)

    print("\nSum Answer:", sum_result.get("answer"))
    print("LLM latency:", sum_result.get("llm_latency_ms"), "ms")
    print("Status:", sum_result.get("workflow_status"))


    # =====================================================
    # 9. TEST - AVERAGE
    # =====================================================

    average_test_state = {
        "question": "What is the average duration of Cholecystectomy surgeries?",
        "query_type": "aggregation",
        "operation": "average",
        "context": "Retrieved Cholecystectomy cases with recorded durations.",
        "analytics_result": {
            "operation": "average",
            "count": 5,
            "average_minutes": 72.4,
            "average_duration": "1 h 12 min",
        },
        "retry_count": 0,
    }

    average_result = answer_agent(average_test_state)

    print("\nAverage Answer:", average_result.get("answer"))
    print("LLM latency:", average_result.get("llm_latency_ms"), "ms")
    print("Status:", average_result.get("workflow_status"))