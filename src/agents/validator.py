# src/agents/validator.py

import time
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import get_llm
from src.prompts import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_PROMPT
from src.agents.state import SurgeryRAGState
from src.config import MAX_RETRIES


# =========================================================
# 1. STRUCTURED VALIDATOR OUTPUT
# =========================================================

RetryAction = Literal[
    "none",
    "rewrite_query",
    "relax_filters",
    "increase_candidates",
]


class ValidatorOutput(BaseModel):
    grounded: bool = Field(description="True if the generated answer is fully supported by retrieved context or trusted analytics.")
    sufficient_context: bool = Field(description="True if enough evidence exists to answer the question.")
    hallucination_detected: bool = Field(description="True if the answer contains unsupported information.")
    retry_required: bool = Field(description="True if another retrieval attempt may improve the answer.")
    retry_action: RetryAction = Field(default="none", description="Action to take if retry is required.")
    reason: str = Field(description="Short explanation for validation decision.")


# =========================================================
# 2. LOAD SHARED LLM
# =========================================================

llm = get_llm()
validator_llm = llm.with_structured_output(ValidatorOutput)


# =========================================================
# 3. VALIDATOR AGENT
# =========================================================

def validator_agent(state: SurgeryRAGState) -> dict:

    question = state.get("question", "").strip()
    context = state.get("context", "").strip()
    answer = state.get("answer", "").strip()
    retry_count = state.get("retry_count", 0)

    query_type = state.get("query_type", "")
    operation = state.get("operation", "none")
    analytics_result = state.get("analytics_result", {})


    # =====================================================
    # 4. BASIC CHECK
    # =====================================================

    if not question:
        return {
            "grounded": False,
            "sufficient_context": False,
            "hallucination_detected": False,
            "retry_required": False,
            "retry_action": "none",
            "validation_reason": "Question is empty.",
            "validator_latency_ms": None,
            "workflow_status": "failed",
            "error": "Question is empty.",
        }


    # =====================================================
    # 5. DETERMINISTIC ANALYTICS VALIDATION
    # =====================================================

    if query_type == "aggregation" and analytics_result:

        valid = False

        if operation == "sum":
            valid = analytics_result.get("total_duration") is not None

        elif operation == "average":
            valid = analytics_result.get("average_duration") is not None

        elif operation == "maximum":
            valid = analytics_result.get("maximum_duration") is not None

        elif operation == "minimum":
            valid = analytics_result.get("minimum_duration") is not None

        elif operation == "count":
            valid = analytics_result.get("count") is not None

        if valid and answer:
            return {
                "grounded": True,
                "sufficient_context": True,
                "hallucination_detected": False,
                "retry_required": False,
                "retry_action": "none",
                "validation_reason": "Answer is supported by deterministic analytics calculated from retrieved surgery-log data.",
                "validator_latency_ms": 0.0,
                "workflow_status": "completed",
                "error": None,
            }


    # =====================================================
    # 6. NO CONTEXT
    # =====================================================

    if not context:
        retry_allowed = retry_count < MAX_RETRIES

        return {
            "grounded": False,
            "sufficient_context": False,
            "hallucination_detected": False,
            "retry_required": retry_allowed,
            "retry_action": "rewrite_query" if retry_allowed else "none",
            "validation_reason": "No retrieved context available.",
            "validator_latency_ms": None,
            "workflow_status": "retrying" if retry_allowed else "completed",
            "error": None,
        }


    # =====================================================
    # 7. NO ANSWER
    # =====================================================

    if not answer:
        retry_allowed = retry_count < MAX_RETRIES

        return {
            "grounded": False,
            "sufficient_context": bool(context),
            "hallucination_detected": False,
            "retry_required": retry_allowed,
            "retry_action": "rewrite_query" if retry_allowed else "none",
            "validation_reason": "No answer was generated.",
            "validator_latency_ms": None,
            "workflow_status": "retrying" if retry_allowed else "completed",
            "error": None,
        }


    # =====================================================
    # 8. BUILD VALIDATION PROMPT
    # =====================================================

    user_prompt = VALIDATOR_USER_PROMPT.format(
        question=question,
        context=context,
        analytics_result=analytics_result,
        answer=answer,
    )

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]


    # =====================================================
    # 9. LLM VALIDATION
    # =====================================================

    try:
        start_time = time.perf_counter()
        result = validator_llm.invoke(messages)
        validator_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        retry_allowed = retry_count < MAX_RETRIES

        retry_required = (
            result.retry_required
            and retry_allowed
            and not result.grounded
        )

        if not retry_allowed:
            retry_required = False

        retry_action = result.retry_action if retry_required else "none"

        if result.grounded:
            workflow_status = "completed"
        elif retry_required:
            workflow_status = "retrying"
        else:
            workflow_status = "completed"

        return {
            "grounded": result.grounded,
            "sufficient_context": result.sufficient_context,
            "hallucination_detected": result.hallucination_detected,
            "retry_required": retry_required,
            "retry_action": retry_action,
            "validation_reason": result.reason,
            "validator_latency_ms": validator_latency_ms,
            "workflow_status": workflow_status,
            "error": None,
        }

    except Exception as exc:
        retry_allowed = retry_count < MAX_RETRIES

        return {
            "grounded": False,
            "sufficient_context": bool(context),
            "hallucination_detected": False,
            "retry_required": retry_allowed,
            "retry_action": "rewrite_query" if retry_allowed else "none",
            "validation_reason": "Validator LLM failed.",
            "validator_latency_ms": None,
            "workflow_status": "retrying" if retry_allowed else "failed",
            "error": f"Validator error: {str(exc)}",
        }


# =========================================================
# 10. LANGGRAPH CONDITIONAL ROUTING
# =========================================================

def route_after_validator(state: SurgeryRAGState) -> str:
    if state.get("retry_required", False):
        return "rewrite"

    return "finish"


# =========================================================
# 11. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    test_state = {
        "question": "What is the total duration of Cholecystectomy surgeries?",
        "query_type": "aggregation",
        "operation": "sum",
        "context": "Retrieved Cholecystectomy surgery cases.",
        "answer": "The total recorded surgery duration across 5 matching cases is 6 h 2 min (362 minutes).",
        "analytics_result": {
            "operation": "sum",
            "count": 5,
            "total_minutes": 362.0,
            "total_duration": "6 h 2 min",
        },
        "retry_count": 0,
    }

    result = validator_agent(test_state)

    print("\nGrounded:", result.get("grounded"))
    print("Sufficient context:", result.get("sufficient_context"))
    print("Hallucination detected:", result.get("hallucination_detected"))
    print("Retry required:", result.get("retry_required"))
    print("Retry action:", result.get("retry_action"))
    print("Reason:", result.get("validation_reason"))
    print("Validator latency:", result.get("validator_latency_ms"), "ms")