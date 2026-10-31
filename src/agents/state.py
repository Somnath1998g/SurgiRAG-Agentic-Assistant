# src/agents/state.py

from typing import TypedDict, Optional, List, Dict, Any


class SurgeryRAGState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Each agent/node reads values from this state
    and updates only the fields it is responsible for.
    """

    # =====================================================
    # 1. ORIGINAL USER INPUT
    # =====================================================

    question: str


    # =====================================================
    # 2. ROUTER OUTPUT
    # =====================================================

    # Examples:
    # case_lookup
    # instrument_lookup
    # timing_duration
    # pedal_activity
    # comparison
    # aggregation
    # unsupported

    query_type: Optional[str]

    # Metadata filters extracted by router.
    # Example:
    #
    # {
    #     "surgery_type": "Cholecystectomy",
    #     "surgeon_name": "Dr MERAI"
    # }

    filters: Dict[str, Any]

    needs_aggregation: bool

    supported: bool


    # =====================================================
    # 3. RETRIEVAL INPUT
    # =====================================================

    # Query actually used for retrieval.
    # Initially same as question.
    # On retry it may contain rewritten query.

    search_query: str


    # =====================================================
    # 4. RETRIEVAL OUTPUT
    # =====================================================

    # Retrieved chunks converted from DataFrame
    # to list[dict] so LangGraph state stays serializable.

    retrieved_chunks: List[Dict[str, Any]]

    # Context string constructed from retrieved chunks.

    context: str

    # Retrieval latency information.
    #
    # Example:
    # {
    #     "faiss_latency_ms": 2.4,
    #     "bm25_latency_ms": 5.1,
    #     "retrieval_latency_ms": 8.3
    # }

    retrieval_timing: Dict[str, float]

    candidate_k: int
    operation: Optional[str]
    analytics_result: Dict[str, Any]
    limit: Optional[int]
    sort_by: Optional[str]
    sort_order: Optional[str]


    # =====================================================
    # 5. ANSWER AGENT OUTPUT
    # =====================================================

    answer: Optional[str]
    llm_latency_ms: Optional[float]


    # =====================================================
    # 6. VALIDATOR OUTPUT
    # =====================================================

    grounded: bool

    sufficient_context: bool

    hallucination_detected: bool

    retry_required: bool

    # Possible values:
    # none
    # rewrite_query
    # relax_filters
    # increase_candidates

    retry_action: Optional[str]

    validation_reason: Optional[str]
    
    validator_latency_ms: Optional[float]


    # =====================================================
    # 7. RETRY CONTROL
    # =====================================================

    retry_count: int

    rewritten_query: Optional[str]


    # =====================================================
    # 8. FINAL OUTPUT / WORKFLOW STATUS
    # =====================================================

    final_answer: Optional[str]

    workflow_status: Optional[str]

    # Example:
    # routed
    # retrieved
    # answered
    # retrying
    # completed
    # unsupported
    # failed


    # =====================================================
    # 9. ERROR HANDLING
    # =====================================================

    error: Optional[str]