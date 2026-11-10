# src/agents/retriever.py

import pandas as pd

from src.retrieval import retrieve, retrieve_all_matching, build_context
from src.agents.state import SurgeryRAGState


# =========================================================
# 1. RETRIEVER AGENT
# =========================================================

def retriever_agent(state: SurgeryRAGState) -> dict:

    search_query = str(state.get("search_query", state.get("question", ""))).strip()
    filters = state.get("filters", {}) or {}
    retry_count = state.get("retry_count", 0)

    query_type = state.get("query_type", "")
    needs_aggregation = state.get("needs_aggregation", False)


    # =====================================================
    # 2. BASIC VALIDATION
    # =====================================================

    if not search_query:
        return {
            "retrieved_chunks": [],
            "context": "",
            "retrieval_timing": {},
            "candidate_k": 0,
            "retrieval_mode": "none",
            "workflow_status": "failed",
            "error": "Search query is empty.",
        }


    # =====================================================
    # 3. CALL RETRIEVAL SERVICE
    # =====================================================

    try:

        # -------------------------------------------------
        # Aggregation → retrieve ALL matching records
        # -------------------------------------------------

        if needs_aggregation or query_type == "aggregation":

            retrieval_output = retrieve_all_matching(
                filters=filters,
            )

            retrieval_mode = "all_matching"

        # -------------------------------------------------
        # Normal RAG → Top-K hybrid retrieval
        # -------------------------------------------------

        else:

            retrieval_output = retrieve(
                query=search_query,
                filters=filters,
                retry_count=retry_count,
            )

            retrieval_mode = "hybrid_top_k"


        # =================================================
        # 4. GET RETRIEVAL OUTPUT
        # =================================================

        results_df = retrieval_output.get("results", pd.DataFrame())
        timing = retrieval_output.get("timing", {})
        candidate_k = retrieval_output.get("candidate_k", 0)


        # =================================================
        # 5. NO RESULTS
        # =================================================

        if results_df is None or results_df.empty:
            return {
                "retrieved_chunks": [],
                "context": "",
                "retrieval_timing": timing,
                "candidate_k": candidate_k,
                "retrieval_mode": retrieval_mode,
                "workflow_status": "retrieved",
                "error": None,
            }


        # =================================================
        # 6. BUILD CONTEXT
        # =================================================

        # For aggregation we may have many records.
        # Analytics uses ALL records, but LLM context does not
        # need hundreds of chunks.
        if retrieval_mode == "all_matching":
            context_df = results_df.head(5)
        else:
            context_df = results_df

        context = build_context(context_df)


        # =================================================
        # 7. SERIALIZE ALL RETRIEVED RECORDS
        # =================================================

        results_df = results_df.where(pd.notnull(results_df), None)
        retrieved_chunks = results_df.to_dict(orient="records")


        # =================================================
        # 8. RETURN STATE UPDATE
        # =================================================

        return {
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "retrieval_timing": timing,
            "candidate_k": candidate_k,
            "retrieval_mode": retrieval_mode,
            "workflow_status": "retrieved",
            "error": None,
        }


    # =====================================================
    # 9. ERROR HANDLING
    # =====================================================

    except Exception as exc:
        return {
            "retrieved_chunks": [],
            "context": "",
            "retrieval_timing": {},
            "candidate_k": 0,
            "retrieval_mode": "failed",
            "workflow_status": "failed",
            "error": f"Retrieval error: {str(exc)}",
        }


# =========================================================
# 10. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    test_states = [

        # Normal Top-K retrieval
        {
            "question": "Which instruments were used during Cholecystectomy?",
            "search_query": "Which instruments were used during Cholecystectomy?",
            "filters": {"surgery_type": "Cholecystectomy"},
            "query_type": "instrument_lookup",
            "needs_aggregation": False,
            "retry_count": 0,
        },

        # All matching retrieval
        {
            "question": "What is the average duration of Cholecystectomy surgeries?",
            "search_query": "average duration of Cholecystectomy surgeries",
            "filters": {"surgery_type": "Cholecystectomy"},
            "query_type": "aggregation",
            "needs_aggregation": True,
            "retry_count": 0,
        },
    ]

    for test_state in test_states:

        result = retriever_agent(test_state)

        print("\n" + "=" * 70)
        print("Question:", test_state["question"])
        print("Retrieval mode:", result.get("retrieval_mode"))
        print("Candidate K:", result.get("candidate_k"))
        print("Retrieved chunks:", len(result.get("retrieved_chunks", [])))
        print("Retrieval timing:", result.get("retrieval_timing"))
        print("Workflow status:", result.get("workflow_status"))
        print("Error:", result.get("error"))