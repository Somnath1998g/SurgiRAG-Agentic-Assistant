# src/agents/retriever.py

import pandas as pd

from src.retrieval import (
    retrieve,
    build_context,
)
from src.agents.state import SurgeryRAGState


# =========================================================
# 1. RETRIEVER AGENT / LANGGRAPH NODE
# =========================================================

def retriever_agent(
    state: SurgeryRAGState,
) -> dict:
    """
    LangGraph retrieval node.

    Reads:
        search_query
        filters
        retry_count

    Calls:
        src.retrieval.retrieve()

    Produces:
        retrieved_chunks
        context
        retrieval_timing
        candidate_k
        workflow_status
    """

    # -----------------------------------------------------
    # 1. Get current search query
    # -----------------------------------------------------

    search_query = state.get(
        "search_query",
        state.get(
            "question",
            "",
        ),
    )

    search_query = str(
        search_query
    ).strip()


    # -----------------------------------------------------
    # 2. Get metadata filters
    # -----------------------------------------------------

    filters = state.get(
        "filters",
        {},
    )

    if filters is None:
        filters = {}


    # -----------------------------------------------------
    # 3. Get retry count
    # -----------------------------------------------------

    retry_count = state.get(
        "retry_count",
        0,
    )


    # -----------------------------------------------------
    # 4. Basic validation
    # -----------------------------------------------------

    if not search_query:

        return {
            "retrieved_chunks": [],
            "context": "",
            "retrieval_timing": {},
            "candidate_k": 0,
            "workflow_status": "failed",
            "error": "Search query is empty.",
        }


    # =====================================================
    # 5. CALL RETRIEVAL SERVICE
    # =====================================================

    try:

        retrieval_output = retrieve(
            query=search_query,
            filters=filters,
            retry_count=retry_count,
        )


        # -----------------------------------------------
        # Get dataframe
        # -----------------------------------------------

        results_df = retrieval_output.get(
            "results",
            pd.DataFrame(),
        )


        # -----------------------------------------------
        # Get timing
        # -----------------------------------------------

        timing = retrieval_output.get(
            "timing",
            {},
        )


        # -----------------------------------------------
        # Candidate pool size
        # -----------------------------------------------

        candidate_k = retrieval_output.get(
            "candidate_k",
            0,
        )


        # =================================================
        # 6. NO RESULTS FOUND
        # =================================================

        if (
            results_df is None
            or results_df.empty
        ):

            return {
                "retrieved_chunks": [],
                "context": "",
                "retrieval_timing": timing,
                "candidate_k": candidate_k,
                "workflow_status": "retrieved",
                "error": None,
            }


        # =================================================
        # 7. BUILD CONTEXT FOR ANSWER AGENT
        # =================================================

        context = build_context(
            results_df
        )


        # =================================================
        # 8. CONVERT DATAFRAME TO SERIALIZABLE RECORDS
        # =================================================

        retrieved_chunks = (
            results_df
            .replace(
                {
                    float("nan"): None
                }
            )
            .to_dict(
                orient="records"
            )
        )


        # =================================================
        # 9. RETURN STATE UPDATE
        # =================================================

        return {
            "retrieved_chunks":
                retrieved_chunks,

            "context":
                context,

            "retrieval_timing":
                timing,

            "candidate_k":
                candidate_k,

            "workflow_status":
                "retrieved",

            "error":
                None,
        }


    # =====================================================
    # 10. ERROR HANDLING
    # =====================================================

    except Exception as exc:

        return {
            "retrieved_chunks": [],
            "context": "",
            "retrieval_timing": {},
            "candidate_k": 0,
            "workflow_status": "failed",
            "error": (
                f"Retrieval error: "
                f"{str(exc)}"
            ),
        }


# =========================================================
# 11. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    test_state = {
        "question":
            (
                "Which instruments were used "
                "during Cholecystectomy?"
            ),

        "search_query":
            (
                "Which instruments were used "
                "during Cholecystectomy?"
            ),

        "filters": {
            "surgery_type":
                "Cholecystectomy"
        },

        "retry_count":
            0,
    }


    result = retriever_agent(
        test_state
    )


    print(
        "\nWorkflow status:",
        result[
            "workflow_status"
        ],
    )


    print(
        "\nCandidate K:",
        result[
            "candidate_k"
        ],
    )


    print(
        "\nRetrieval timing:"
    )

    print(
        result[
            "retrieval_timing"
        ]
    )


    print(
        "\nRetrieved chunks:",
        len(
            result[
                "retrieved_chunks"
            ]
        ),
    )


    print(
        "\nContext:\n"
    )

    print(
        result[
            "context"
        ][:2000]
    )