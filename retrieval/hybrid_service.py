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
from rag.retriever import retrieve
from structured_query.engine import answer_structured_query
from structured_query.parser import parse_query, to_display_sql
from retrieval.merger import merge_results
from schema.variables import has_variable_data, VARIABLE_LABELS, DEFAULT_VARIABLE
import logging

load_dotenv()


def hybrid_answer(question: str, history: list = None) -> dict:
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
    parsed = parse_query(question)
    variable = parsed.variable if parsed.variable else "temp_c"
    semantic_rows = retrieve(question, top_k=5, distance_threshold=threshold, parsed_query=parsed, variable=variable)

    # 3. Deduplicate and merge rows
    combined_df = merge_results(struct_rows, semantic_rows)
    if variable != DEFAULT_VARIABLE and not has_variable_data(combined_df, variable):
        var_label = VARIABLE_LABELS.get(variable, variable)
        metadata["error"] = "variable_unavailable"
        metadata["variable"] = variable
        return {
            "summary": f"No {var_label} measurements were found among the retrieved observations for this query.",
            "rows": pd.DataFrame(),
            "metadata": metadata,
            "sql": "-- Short-circuited: No data for requested variable in retrieved rows"
        }

    # 4. Build Context-Fused Prompt
    # We pass only the semantic rows as "Supporting Records" so the LLM doesn't double-count
    # the structured data, but we return combined_df for the UI visualisations.
    prompt = build_hybrid_prompt(question, struct_summary, semantic_rows, variable)

    # Calculate hybrid confidence based on retrieval success
    # If we have structured rows, we are confident. If we lack semantic context, we just lower it a bit.
    confidence = 0.90
    if semantic_rows.empty:
        confidence = 0.70  # Lower confidence since we couldn't find a semantic explanation

    # 5. Invoke LLM and generate SQL
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        
        messages = history.copy() if history else []
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        final_answer = response.choices[0].message.content.strip()
    except Exception as e:
        logging.getLogger(__name__).error(f"LLM call failed: {e}")
        final_answer = f"{struct_summary}\n\n(LLM narrative unavailable)"

    if combined_df.empty:
        sql = "-- No data found"
    else:
        sql = to_display_sql(parsed)

    return {
        "summary": final_answer,
        "rows": combined_df,
        "metadata": metadata,
        "sql": sql,
        "confidence": confidence
    }
