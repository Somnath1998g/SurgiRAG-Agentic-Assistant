# src/agents/rewriter.py

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from src.llm import get_llm
from src.prompts import (
    REWRITER_SYSTEM_PROMPT,
    REWRITER_USER_PROMPT,
)
from src.config import MAX_RETRIES
from src.agents.state import SurgeryRAGState


# =========================================================
# 1. LOAD SHARED LLM
# =========================================================

llm = get_llm()


# =========================================================
# 2. RELAX METADATA FILTERS
# =========================================================

def relax_metadata_filters(
    filters: dict,
) -> dict:
    """
    Remove one potentially restrictive metadata filter.

    Priority:
        chunk_type
        surgeon_name
        surgery_type

    Strong identifiers such as case_id and source_file
    are preserved because removing them may change
    the user's intended case.
    """

    if not filters:
        return {}

    relaxed = filters.copy()

    removable_priority = [
        "chunk_type",
        "surgeon_name",
        "surgery_type",
    ]

    for key in removable_priority:

        if key in relaxed:

            relaxed.pop(key)

            return relaxed

    # Nothing safe to remove.
    return relaxed


# =========================================================
# 3. QUERY REWRITE USING LLM
# =========================================================

def rewrite_search_query(
    question: str,
    validation_reason: str,
    filters: dict,
) -> str:
    """
    Rewrite the user's query for better FAISS + BM25
    retrieval while preserving the original intent.
    """

    user_prompt = REWRITER_USER_PROMPT.format(
        question=question,
        validation_reason=validation_reason,
        filters=filters,
    )

    messages = [
        SystemMessage(
            content=REWRITER_SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_prompt
        ),
    ]

    response = llm.invoke(
        messages
    )

    rewritten_query = getattr(
        response,
        "content",
        "",
    )

    rewritten_query = str(
        rewritten_query
    ).strip()

    # Safe fallback
    if not rewritten_query:
        rewritten_query = question

    return rewritten_query


# =========================================================
# 4. REWRITER AGENT
# =========================================================

def rewriter_agent(
    state: SurgeryRAGState,
) -> dict:
    """
    LangGraph retry/rewrite node.

    Reads:
        question
        filters
        retry_action
        validation_reason
        retry_count

    Updates:
        search_query
        rewritten_query
        filters
        retry_count
        workflow_status

    Maximum retries are controlled by MAX_RETRIES.
    """

    question = state.get(
        "question",
        "",
    ).strip()

    current_search_query = state.get(
        "search_query",
        question,
    ).strip()

    filters = state.get(
        "filters",
        {},
    ).copy()

    retry_action = state.get(
        "retry_action",
        "rewrite_query",
    )

    validation_reason = state.get(
        "validation_reason",
        "",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )


    # =====================================================
    # 5. CHECK RETRY LIMIT
    # =====================================================

    if retry_count >= MAX_RETRIES:

        return {
            "retry_required": False,
            "retry_action": "none",
            "workflow_status": "completed",
            "error": None,
        }


    # Next attempt
    new_retry_count = (
        retry_count + 1
    )


    # =====================================================
    # 6. ACTION: REWRITE QUERY
    # =====================================================

    if retry_action == "rewrite_query":

        try:

            rewritten_query = (
                rewrite_search_query(
                    question=question,
                    validation_reason=(
                        validation_reason
                    ),
                    filters=filters,
                )
            )

            return {
                "search_query":
                    rewritten_query,

                "rewritten_query":
                    rewritten_query,

                "filters":
                    filters,

                "retry_count":
                    new_retry_count,

                "retry_required":
                    False,

                "workflow_status":
                    "retrying",

                "error":
                    None,
            }

        except Exception as exc:

            # If rewriting fails, use original
            # search query rather than crashing graph.

            return {
                "search_query":
                    current_search_query,

                "rewritten_query":
                    current_search_query,

                "filters":
                    filters,

                "retry_count":
                    new_retry_count,

                "retry_required":
                    False,

                "workflow_status":
                    "retrying",

                "error":
                    (
                        "Query rewriting failed: "
                        f"{str(exc)}"
                    ),
            }


    # =====================================================
    # 7. ACTION: RELAX FILTERS
    # =====================================================

    if retry_action == "relax_filters":

        relaxed_filters = (
            relax_metadata_filters(
                filters
            )
        )

        return {
            "search_query":
                current_search_query,

            "rewritten_query":
                None,

            "filters":
                relaxed_filters,

            "retry_count":
                new_retry_count,

            "retry_required":
                False,

            "workflow_status":
                "retrying",

            "error":
                None,
        }


    # =====================================================
    # 8. ACTION: INCREASE CANDIDATES
    # =====================================================

    if retry_action == "increase_candidates":

        # Nothing else needs changing here.
        #
        # src/retrieval.py already checks:
        #
        # if retry_count > 0:
        #     candidate_k = RETRY_CANDIDATE_K
        #
        # Therefore incrementing retry_count is enough.

        return {
            "search_query":
                current_search_query,

            "rewritten_query":
                None,

            "filters":
                filters,

            "retry_count":
                new_retry_count,

            "retry_required":
                False,

            "workflow_status":
                "retrying",

            "error":
                None,
        }


    # =====================================================
    # 9. FALLBACK
    # =====================================================

    # If validator gives an unknown action,
    # default to query rewriting.

    try:

        rewritten_query = (
            rewrite_search_query(
                question=question,
                validation_reason=(
                    validation_reason
                ),
                filters=filters,
            )
        )

    except Exception:

        rewritten_query = (
            current_search_query
        )


    return {
        "search_query":
            rewritten_query,

        "rewritten_query":
            rewritten_query,

        "filters":
            filters,

        "retry_count":
            new_retry_count,

        "retry_required":
            False,

        "workflow_status":
            "retrying",

        "error":
            None,
    }


# =========================================================
# 10. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------
    # Test 1: rewrite query
    # -----------------------------------------

    test_state_1 = {

        "question":
            (
                "What instruments did MERAI "
                "use in that gallbladder surgery?"
            ),

        "search_query":
            (
                "What instruments did MERAI "
                "use in that gallbladder surgery?"
            ),

        "filters": {
            "surgeon_name": "MERAI",
        },

        "validation_reason":
            (
                "Retrieved context did not contain "
                "enough instrument information."
            ),

        "retry_action":
            "rewrite_query",

        "retry_count":
            0,
    }


    result_1 = rewriter_agent(
        test_state_1
    )

    print(
        "\n=== REWRITE QUERY TEST ==="
    )

    print(
        "Original:",
        test_state_1[
            "search_query"
        ],
    )

    print(
        "Rewritten:",
        result_1[
            "search_query"
        ],
    )

    print(
        "Retry count:",
        result_1[
            "retry_count"
        ],
    )


    # -----------------------------------------
    # Test 2: relax filters
    # -----------------------------------------

    test_state_2 = {

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
                "Cholecystectomy",

            "surgeon_name":
                "Dr MERAI",

            "chunk_type":
                "instrument_summary",
        },

        "validation_reason":
            "Metadata filters may be too restrictive.",

        "retry_action":
            "relax_filters",

        "retry_count":
            0,
    }


    result_2 = rewriter_agent(
        test_state_2
    )

    print(
        "\n=== RELAX FILTER TEST ==="
    )

    print(
        "Old filters:",
        test_state_2[
            "filters"
        ],
    )

    print(
        "New filters:",
        result_2[
            "filters"
        ],
    )


    # -----------------------------------------
    # Test 3: increase candidates
    # -----------------------------------------

    test_state_3 = {

        "question":
            "Show instrument usage.",

        "search_query":
            "Show instrument usage.",

        "filters": {},

        "validation_reason":
            (
                "Relevant chunks were found "
                "but context was insufficient."
            ),

        "retry_action":
            "increase_candidates",

        "retry_count":
            0,
    }


    result_3 = rewriter_agent(
        test_state_3
    )

    print(
        "\n=== INCREASE CANDIDATES TEST ==="
    )

    print(
        "Retry count:",
        result_3[
            "retry_count"
        ],
    )

    print(
        "Search query:",
        result_3[
            "search_query"
        ],
    )