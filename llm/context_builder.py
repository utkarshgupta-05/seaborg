"""
llm/context_builder.py

Builds the prompt for the HYBRID path.
Fuses structured data (authoritative numerical facts) with semantic data (contextual rows)
to provide a single, grounded prompt for the LLM.
"""

import pandas as pd

HYBRID_PROMPT = """You are SeaBorg, a strict ocean data analyst.

You have been provided with two types of evidence to answer the user's question:
1. AUTHORITATIVE DATA SUMMARY: This contains exact aggregations (averages, counts, ranges) calculated directly from the database. You MUST use these numbers when referring to overall statistics. Do not attempt to recalculate them from the sample rows.
2. SUPPORTING RECORDS: A small semantic sample of related float profiles. Use these to explain patterns, variations, or provide context, but NEVER contradict the authoritative summary.

RULES:
1. Answer ONLY using the provided evidence. Never use outside knowledge.
2. If the user asks for a statistic (e.g. "average temperature"), quote the Authoritative Data Summary.
3. If the user asks "why" or asks for an explanation, look at the Supporting Records to provide oceanographic context based ONLY on the data shown.
4. Do NOT hallucinate. Do NOT infer geographic regions if not stated.

AUTHORITATIVE DATA SUMMARY:
{structured_summary}

SUPPORTING RECORDS:
{semantic_context}

Question: {question}

Answer:"""

from schema.variables import DEFAULT_VARIABLE
from .formatters import format_row

def build_hybrid_prompt(question: str, structured_summary: str, semantic_rows: pd.DataFrame, variable: str = DEFAULT_VARIABLE) -> str:
    """
    Constructs a grounded prompt containing both authoritative aggregated facts
    and specific supporting rows for context.
    """
    if semantic_rows is None or semantic_rows.empty:
        context = "No supporting records retrieved."
    else:
        # Cap at 5 rows to prevent context bloat
        limited = semantic_rows.head(5)
        bullets = []
        for _, row in limited.iterrows():
            bullets.append(format_row(row, variable))
        context = "\n".join(bullets) if bullets else "No supporting records retrieved."
        
    return HYBRID_PROMPT.format(
        structured_summary=structured_summary,
        semantic_context=context,
        question=question
    )
