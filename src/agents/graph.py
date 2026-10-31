# src/agents/graph.py

from langgraph.graph import StateGraph, START, END

from src.agents.state import SurgeryRAGState
from src.agents.router import router_agent, route_after_router
from src.agents.retriever import retriever_agent
from src.agents.answer import answer_agent
from src.agents.validator import validator_agent, route_after_validator
from src.agents.rewriter import rewriter_agent
from src.agents.analytics import analytics_agent

from src.config import FALLBACK_RESPONSE
from src.prompts import UNSUPPORTED_RESPONSE


# =========================================================
# 1. UNSUPPORTED QUERY NODE
# =========================================================

def unsupported_node(state: SurgeryRAGState) -> dict:
    return {
        "answer": UNSUPPORTED_RESPONSE,
        "final_answer": UNSUPPORTED_RESPONSE,
        "retry_required": False,
        "workflow_status": "unsupported",
        "error": None,
    }


# =========================================================
# 2. FINAL RESPONSE NODE
# =========================================================

def finish_node(state: SurgeryRAGState) -> dict:

    grounded = state.get("grounded", False)
    answer = state.get("answer", "")
    sufficient_context = state.get("sufficient_context", False)
    hallucination_detected = state.get("hallucination_detected", False)

    if grounded and sufficient_context and not hallucination_detected and answer:
        final_answer = answer
    else:
        final_answer = FALLBACK_RESPONSE

    return {
        "final_answer": final_answer,
        "workflow_status": "completed",
        "retry_required": False,
    }


# =========================================================
# 3. BUILD LANGGRAPH
# =========================================================

def build_surgery_rag_graph():

    builder = StateGraph(SurgeryRAGState)

    # Nodes
    builder.add_node("router", router_agent)
    builder.add_node("retrieval", retriever_agent)
    builder.add_node("analytics", analytics_agent)
    builder.add_node("answer", answer_agent)
    builder.add_node("validator", validator_agent)
    builder.add_node("rewrite", rewriter_agent)
    builder.add_node("finish", finish_node)
    builder.add_node("unsupported", unsupported_node)

    # START → Router
    builder.add_edge(START, "router")

    # Router → Retrieval / Unsupported
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieval": "retrieval",
            "unsupported": "unsupported",
        },
    )

    # Retrieval → Analytics
    builder.add_edge("retrieval","analytics")

    # Analytics → Answer
    builder.add_edge("analytics", "answer")

    # Answer → Validator
    builder.add_edge("answer", "validator")

    # Validator → Rewrite / Finish
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "rewrite": "rewrite",
            "finish": "finish",
        },
    )

    # Retry flow
    builder.add_edge("rewrite", "retrieval")

    # Final edges
    builder.add_edge("finish", END)
    builder.add_edge("unsupported", END)

    return builder.compile()


# =========================================================
# 4. CREATE SHARED GRAPH INSTANCE
# =========================================================

surgery_agent = build_surgery_rag_graph()


# =========================================================
# 5. HELPER FUNCTION
# =========================================================

def ask_surgery_rag(question: str):

    initial_state = {
        "question": question,
        "search_query": question,
        "filters": {},
        "retry_count": 0,
        "retry_required": False,
        "workflow_status": "started",
        "error": None,
    }

    result = surgery_agent.invoke(initial_state)

    return result


# =========================================================
# 6. SIMPLE TEST + DEBUGGING
# =========================================================

if __name__ == "__main__":

    test_questions = [
        # "Which instruments were used during Cholecystectomy?",
        # "What was the total surgery duration for the Cholecystectomy case?",
        "What is the total duration of the last two cases for the Cholecystectomy case?",
        # "What is the weather today?",
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)

        result = ask_surgery_rag(question)

        print("\nQUERY TYPE:", result.get("query_type"))
        print("FILTERS:", result.get("filters"))
        print("SEARCH QUERY:", result.get("search_query"))
        print("OPERATION:", result.get("operation"))
        print("ANALYTICS RESULT:", result.get("analytics_result"))

        print("\nRETRIEVED CHUNKS:")
        

        for x in result.get("retrieved_chunks", []):
            print("Chunk Type:", x.get("chunk_type"))
            print("Case ID:", x.get("case_id"))
            print("Source File:", x.get("source_file"))
            print("Text:", x.get("text"))
            print("-" * 50)

        print("\nCONTEXT:", result.get("context"))

        print("\nANSWER:", result.get("answer"))

        print("\nGROUNDED:", result.get("grounded"))
        print("SUFFICIENT CONTEXT:", result.get("sufficient_context"))
        print("HALLUCINATION:", result.get("hallucination_detected"))

        print("\nVALIDATION REASON:", result.get("validation_reason"))
        print("RETRY REQUIRED:", result.get("retry_required"))
        print("RETRY ACTION:", result.get("retry_action"))
        print("RETRY COUNT:", result.get("retry_count"))

        print("\nFINAL ANSWER:", result.get("final_answer"))

        print("\nWORKFLOW STATUS:", result.get("workflow_status"))
        print("RETRIEVAL TIMING:", result.get("retrieval_timing"))
        print("LLM LATENCY:", result.get("llm_latency_ms"))
        print("VALIDATOR LATENCY:", result.get("validator_latency_ms"))

        print("\nERROR:", result.get("error"))