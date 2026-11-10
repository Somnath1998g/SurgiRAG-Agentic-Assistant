# src/retrieval.py

import re
import time

import faiss
import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    FAISS_INDEX_PATH,
    VECTOR_METADATA_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    CANDIDATE_K,
    RETRY_CANDIDATE_K,
    RRF_K,
)


# =========================================================
# 1. LOAD VECTOR STORE / METADATA
# =========================================================

if not FAISS_INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {FAISS_INDEX_PATH}"
    )

if not VECTOR_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {VECTOR_METADATA_PATH}"
    )


faiss_index = faiss.read_index(
    str(FAISS_INDEX_PATH)
)

metadata_df = pd.read_csv(
    VECTOR_METADATA_PATH
)


# Optional but strongly recommended for filtered FAISS search.
if EMBEDDINGS_PATH.exists():
    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype("float32")
else:
    embeddings = None


print(
    f"Loaded {len(metadata_df)} metadata rows "
    f"and {faiss_index.ntotal} FAISS vectors."
)


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# =========================================================
# 3. CREATE BM25 SEARCH TEXT
# =========================================================

def safe_column(df, column):
    if column in df.columns:
        return (
            df[column]
            .fillna("")
            .astype(str)
        )

    return pd.Series(
        [""] * len(df),
        index=df.index
    )


if "search_text" not in metadata_df.columns:

    metadata_df["search_text"] = (
        safe_column(metadata_df, "case_id")
        + " "
        + safe_column(metadata_df, "source_file")
        + " "
        + safe_column(metadata_df, "surgery_type")
        + " "
        + safe_column(metadata_df, "surgeon_name")
        + " "
        + safe_column(metadata_df, "chunk_type")
        + " "
        + safe_column(metadata_df, "instrument_names")
        + " "
        + safe_column(metadata_df, "text")
    )


# =========================================================
# 4. TOKENIZER
# =========================================================

def tokenize(text):
    """
    Basic tokenizer for BM25.

    Keeps:
    - letters
    - numbers
    - underscores
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9_]+",
        " ",
        text
    )

    return text.split()


# =========================================================
# 5. GLOBAL BM25 INDEX
# =========================================================

tokenized_corpus = [
    tokenize(text)
    for text in metadata_df[
        "search_text"
    ].tolist()
]

bm25_index = BM25Okapi(
    tokenized_corpus
)


# =========================================================
# 6. METADATA FILTERING
# =========================================================

def apply_metadata_filters(
    df,
    filters=None,
):
    """
    Filter metadata before retrieval.

    Example:

    filters = {
        "surgery_type": "Cholecystectomy",
        "surgeon_name": "MERAI"
    }

    String filtering is case-insensitive and supports
    partial matching.
    """

    if not filters:
        return df.copy()

    filtered = df.copy()

    for column, value in filters.items():

        if (
            value is None
            or column not in filtered.columns
        ):
            continue

        # Multiple accepted values
        if isinstance(value, list):

            pattern = "|".join(
                re.escape(str(v))
                for v in value
            )

            filtered = filtered[
                filtered[column]
                .fillna("")
                .astype(str)
                .str.contains(
                    pattern,
                    case=False,
                    na=False,
                    regex=True,
                )
            ]

        # String / scalar
        else:

            filtered = filtered[
                filtered[column]
                .fillna("")
                .astype(str)
                .str.contains(
                    re.escape(str(value)),
                    case=False,
                    na=False,
                    regex=True,
                )
            ]

    return filtered.copy()


# =========================================================
# 7. QUERY EMBEDDING
# =========================================================

def encode_query(query):
    """
    Encode query and L2-normalize it.

    Your FAISS index uses normalized embeddings +
    inner product, which approximates cosine similarity.
    """

    vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(vector)

    return vector


# =========================================================
# 8. FAISS SEMANTIC SEARCH
# =========================================================

def semantic_search(
    query,
    top_k=CANDIDATE_K,
    filters=None,
):
    start = time.perf_counter()

    query_vector = encode_query(query)

    # -----------------------------------------------------
    # No metadata filters
    # -----------------------------------------------------

    if not filters:

        search_k = min(
            top_k,
            faiss_index.ntotal
        )

        scores, indices = faiss_index.search(
            query_vector,
            search_k
        )

        rows = []

        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):

            if idx < 0:
                continue

            row = metadata_df.iloc[
                int(idx)
            ].to_dict()

            row["semantic_rank"] = rank
            row["semantic_score"] = float(
                score
            )

            rows.append(row)

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return (
            pd.DataFrame(rows),
            latency_ms,
        )

    # -----------------------------------------------------
    # Metadata-filtered semantic search
    # -----------------------------------------------------

    filtered_df = apply_metadata_filters(
        metadata_df,
        filters,
    )

    if filtered_df.empty:

        return (
            pd.DataFrame(),
            (
                time.perf_counter()
                - start
            )
            * 1000,
        )

    # Best method:
    # use saved embeddings and create small temporary index
    if embeddings is not None:

        original_indices = (
            filtered_df.index
            .to_numpy()
        )

        filtered_embeddings = embeddings[
            original_indices
        ].copy()

        temp_index = faiss.IndexFlatIP(
            filtered_embeddings.shape[1]
        )

        temp_index.add(
            filtered_embeddings
        )

        search_k = min(
            top_k,
            len(filtered_df)
        )

        scores, local_indices = (
            temp_index.search(
                query_vector,
                search_k,
            )
        )

        rows = []

        for rank, (
            local_idx,
            score,
        ) in enumerate(
            zip(
                local_indices[0],
                scores[0],
            ),
            start=1,
        ):

            if local_idx < 0:
                continue

            original_idx = (
                original_indices[
                    int(local_idx)
                ]
            )

            row = metadata_df.loc[
                original_idx
            ].to_dict()

            row["semantic_rank"] = rank
            row["semantic_score"] = float(
                score
            )

            rows.append(row)

    # Fallback:
    # retrieve larger pool and filter afterward
    else:

        search_k = min(
            max(top_k * 10, 100),
            faiss_index.ntotal,
        )

        scores, indices = faiss_index.search(
            query_vector,
            search_k,
        )

        allowed_indices = set(
            filtered_df.index.tolist()
        )

        rows = []

        rank = 1

        for idx, score in zip(
            indices[0],
            scores[0],
        ):

            if idx < 0:
                continue

            if int(idx) not in allowed_indices:
                continue

            row = metadata_df.iloc[
                int(idx)
            ].to_dict()

            row["semantic_rank"] = rank
            row["semantic_score"] = float(
                score
            )

            rows.append(row)

            rank += 1

            if len(rows) >= top_k:
                break

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        pd.DataFrame(rows),
        latency_ms,
    )


# =========================================================
# 9. BM25 KEYWORD SEARCH
# =========================================================

def keyword_search(
    query,
    top_k=CANDIDATE_K,
    filters=None,
):
    start = time.perf_counter()

    filtered_df = apply_metadata_filters(
        metadata_df,
        filters,
    )

    if filtered_df.empty:

        return (
            pd.DataFrame(),
            (
                time.perf_counter()
                - start
            )
            * 1000,
        )

    query_tokens = tokenize(query)

    # -----------------------------------------------------
    # Use global BM25 if no filtering
    # -----------------------------------------------------

    if not filters:

        scores = bm25_index.get_scores(
            query_tokens
        )

        working_df = (
            metadata_df.copy()
        )

    # -----------------------------------------------------
    # Build temporary BM25 over filtered subset
    # -----------------------------------------------------

    else:

        filtered_corpus = [
            tokenize(text)
            for text in filtered_df[
                "search_text"
            ].tolist()
        ]

        temp_bm25 = BM25Okapi(
            filtered_corpus
        )

        scores = temp_bm25.get_scores(
            query_tokens
        )

        working_df = (
            filtered_df.copy()
        )

    working_df["keyword_score"] = (
        scores
    )

    working_df = (
        working_df
        .sort_values(
            "keyword_score",
            ascending=False,
        )
        .head(top_k)
    )

    rows = []

    for rank, (_, row) in enumerate(
        working_df.iterrows(),
        start=1,
    ):

        result = row.to_dict()

        result["keyword_rank"] = rank

        result["keyword_score"] = float(
            result["keyword_score"]
        )

        rows.append(result)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        pd.DataFrame(rows),
        latency_ms,
    )


# =========================================================
# 10. UNIQUE DOCUMENT KEY
# =========================================================

def get_chunk_key(row):
    """
    Prefer chunk_id.

    Fallback to combination of source + chunk type + text.
    """

    chunk_id = row.get(
        "chunk_id"
    )

    if (
        chunk_id is not None
        and str(chunk_id) != "nan"
    ):
        return str(chunk_id)

    return (
        str(row.get("source_file", ""))
        + "_"
        + str(row.get("chunk_type", ""))
        + "_"
        + str(row.get("text", ""))[:50]
    )


# =========================================================
# 11. RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    semantic_df,
    keyword_df,
    top_k=TOP_K,
    rrf_k=RRF_K,
):
    """
    RRF formula:

    score += 1 / (rrf_k + rank)

    Raw FAISS and BM25 scores are not directly mixed.
    """

    fused = {}

    # -----------------------------------------------------
    # Semantic results
    # -----------------------------------------------------

    for _, row in semantic_df.iterrows():

        key = get_chunk_key(row)

        rank = int(
            row["semantic_rank"]
        )

        if key not in fused:

            fused[key] = {
                "row": row.to_dict(),
                "rrf_score": 0.0,
                "semantic_rank": None,
                "keyword_rank": None,
            }

        fused[key]["rrf_score"] += (
            1.0
            / (
                rrf_k
                + rank
            )
        )

        fused[key]["semantic_rank"] = rank

    # -----------------------------------------------------
    # BM25 results
    # -----------------------------------------------------

    for _, row in keyword_df.iterrows():

        key = get_chunk_key(row)

        rank = int(
            row["keyword_rank"]
        )

        if key not in fused:

            fused[key] = {
                "row": row.to_dict(),
                "rrf_score": 0.0,
                "semantic_rank": None,
                "keyword_rank": None,
            }

        fused[key]["rrf_score"] += (
            1.0
            / (
                rrf_k
                + rank
            )
        )

        fused[key]["keyword_rank"] = rank

        # Update missing metadata from BM25 row
        fused[key]["row"].update(
            row.to_dict()
        )

    final_rows = []

    for item in fused.values():

        row = item["row"]

        row["rrf_score"] = (
            item["rrf_score"]
        )

        row["semantic_rank"] = (
            item["semantic_rank"]
        )

        row["keyword_rank"] = (
            item["keyword_rank"]
        )

        final_rows.append(row)

    if not final_rows:
        return pd.DataFrame()

    final_df = pd.DataFrame(
        final_rows
    )

    final_df = (
        final_df
        .sort_values(
            "rrf_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    final_df["final_rank"] = (
        np.arange(
            1,
            len(final_df) + 1
        )
    )

    return final_df


# =========================================================
# 12. COMPLETE HYBRID RETRIEVAL
# =========================================================

def hybrid_search(
    query,
    filters=None,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
):
    """
    Full retrieval:

    metadata filtering
        ↓
    FAISS semantic retrieval
        +
    BM25 keyword retrieval
        ↓
    RRF
        ↓
    final top-k
    """

    total_start = time.perf_counter()

    semantic_df, faiss_latency_ms = (
        semantic_search(
            query=query,
            top_k=candidate_k,
            filters=filters,
        )
    )

    keyword_df, bm25_latency_ms = (
        keyword_search(
            query=query,
            top_k=candidate_k,
            filters=filters,
        )
    )

    final_df = reciprocal_rank_fusion(
        semantic_df=semantic_df,
        keyword_df=keyword_df,
        top_k=top_k,
        rrf_k=RRF_K,
    )

    retrieval_latency_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    timing = {
        "faiss_latency_ms": round(
            faiss_latency_ms,
            2,
        ),
        "bm25_latency_ms": round(
            bm25_latency_ms,
            2,
        ),
        "retrieval_latency_ms": round(
            retrieval_latency_ms,
            2,
        ),
    }

    return final_df, timing


# =========================================================
# 13. AGENT-FRIENDLY RETRIEVAL FUNCTION
# =========================================================
def retrieve_all_matching(filters=None):
    """
    Retrieve all chunks matching metadata filters.

    Used for deterministic aggregation queries where top_k
    retrieval would produce incomplete calculations.
    """

    filtered_df = apply_metadata_filters(
        metadata_df,
        filters,
    )

    if filtered_df.empty:
        return {
            "results": filtered_df,
            "timing": {},
            "candidate_k": len(filtered_df),
        }

    return {
        "results": filtered_df.reset_index(drop=True),
        "timing": {
            "retrieval_latency_ms": 0.0,
        },
        "candidate_k": len(filtered_df),
    }

def retrieve(
    query,
    filters=None,
    retry_count=0,
    top_k=TOP_K,
):
    """
    Main function used by LangGraph Retrieval Agent.

    On retry:
    candidate pool is increased automatically.
    """

    if retry_count > 0:
        candidate_k = RETRY_CANDIDATE_K
    else:
        candidate_k = CANDIDATE_K

    results, timing = hybrid_search(
        query=query,
        filters=filters,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    return {
        "results": results,
        "timing": timing,
        "candidate_k": candidate_k,
    }


# =========================================================
# 14. BUILD LLM CONTEXT
# =========================================================

def build_context(
    retrieved_df,
    max_chunks=TOP_K,
):
    """
    Convert retrieved chunks into context
    for Answer Agent.
    """

    if retrieved_df is None:
        return ""

    if retrieved_df.empty:
        return ""

    context_parts = []

    for i, (_, row) in enumerate(
        retrieved_df
        .head(max_chunks)
        .iterrows(),
        start=1,
    ):

        context_parts.append(
            f"""
Retrieved Chunk {i}

Case ID:
{row.get("case_id", "")}

Surgery Type:
{row.get("surgery_type", "")}

Surgeon:
{row.get("surgeon_name", "")}

Chunk Type:
{row.get("chunk_type", "")}

Source File:
{row.get("source_file", "")}

Content:
{row.get("text", "")}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# =========================================================
# 15. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    question = (
        "Which instruments were used "
        "during Cholecystectomy?"
    )

    output = retrieve(
        query=question,
        filters={
            "surgery_type":
            "Cholecystectomy"
        },
    )

    print(
        "\nRetrieval timing:"
    )

    print(
        output["timing"]
    )

    results = output[
        "results"
    ]

    if results.empty:

        print(
            "\nNo matching chunks found."
        )

    else:

        columns = [
            col
            for col in [
                "final_rank",
                "rrf_score",
                "semantic_rank",
                "keyword_rank",
                "case_id",
                "chunk_type",
                "source_file",
            ]
            if col in results.columns
        ]

        print(
            "\nRetrieved results:"
        )

        print(
            results[
                columns
            ].to_string(
                index=False
            )
        )