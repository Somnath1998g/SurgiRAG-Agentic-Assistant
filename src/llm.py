# src/llm.py

from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)


# =========================================================
# 1. CREATE SHARED LLM INSTANCE
# =========================================================

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )


llm = ChatGroq(
    model=LLM_MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=LLM_TEMPERATURE,
)


# =========================================================
# 2. HELPER FUNCTION
# =========================================================

def get_llm():
    """
    Return the shared LLM instance.

    Router, Answer, Validator, and Rewriter agents
    can all reuse this same model.
    """
    return llm


# =========================================================
# 3. SIMPLE TEST
# =========================================================

if __name__ == "__main__":
    print("LLM model:", LLM_MODEL_NAME)

    response = llm.invoke(
        "Reply only with: LLM connection successful"
    )

    print(response.content)