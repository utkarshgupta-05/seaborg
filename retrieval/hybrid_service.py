"""
retrieval/hybrid_service.py

Orchestrates the HYBRID retrieval path.
1. Calls structured service for authoritative facts.
2. Calls semantic retriever for context rows.
3. Deduplicates and merges row data.
4. Builds a grounded hybrid prompt.
5. Invokes the LLM to generate the final narrated answer.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from llm.context_builder import build_hybrid_prompt
from llm.nl_to_sql import generate_sql
from rag.retriever import retrieve
from structured_query.engine import answer_structured_query

load_dotenv()


def hybrid_answer(question: str) -> dict:
    """
    Executes the true hybrid retrieval path.

    Returns a dictionary compatible with the expected ChatResponse:
    {
        "summary": str (the final LLM answer),
        "rows": pd.DataFrame (merged & deduplicated),
        "metadata": dict,
        "sql": str (fallback SQL generation if applicable)
    }
    """
    # 1. Structured Retrieval (Authoritative Facts)
    struct_result = answer_structured_query(question)
    struct_summary = struct_result["summary"]
    struct_rows = struct_result["rows"]
    metadata = struct_result.get("metadata", {})
    metadata["query_type"] = "hybrid"

    # 2. Semantic Retrieval (Context / Explanations)
    # Filter weak matches out using a distance threshold (lower is better for L2 distance)
    # threshold 1.5 is typically safe for MiniLM normalized embeddings
    threshold = float(os.getenv("FAISS_DISTANCE_THRESHOLD", "1.5"))
    semantic_rows = retrieve(question, top_k=5, distance_threshold=threshold)

    # 3. Deduplicate and merge rows
    # Prefer structured rows (authoritative) by placing them first and keeping 'first' duplicate
    combined_df = pd.concat([struct_rows, semantic_rows], ignore_index=True)
    if not combined_df.empty:
        # Ensure we have the subset columns before deduplicating
        subset_cols = [c for c in ["float_id", "date", "depth_m"] if c in combined_df.columns]
        if subset_cols:
            combined_df = combined_df.drop_duplicates(subset=subset_cols, keep="first")

    # 4. Build Context-Fused Prompt
    # We pass only the semantic rows as "Supporting Records" so the LLM doesn't double-count
    # the structured data, but we return combined_df for the UI visualisations.
    prompt = build_hybrid_prompt(question, struct_summary, semantic_rows)

    # Calculate hybrid confidence based on retrieval success
    # If we have structured rows, we are confident. If we lack semantic context, we just lower it a bit.
    confidence = 0.90
    if semantic_rows.empty:
        confidence = 0.70  # Lower confidence since we couldn't find a semantic explanation

    # 5. Invoke LLM
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        final_answer = response.choices[0].message.content.strip()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"LLM call failed: {e}")
        # Fallback if LLM fails
        final_answer = f"{struct_summary}\n\n(LLM narrative unavailable)"

    # Best-effort SQL generation for the UI display if data exists
    if combined_df.empty:
        sql = "-- No data found"
    else:
        try:
            sql = generate_sql(question)
        except Exception:
            sql = "-- SQL generation failed"

    return {
        "summary": final_answer,
        "rows": combined_df,
        "metadata": metadata,
        "sql": sql,
        "confidence": confidence
    }
