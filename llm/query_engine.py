import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from .prompts import build_prompt

load_dotenv()


from structured_query.parser import parse_query, to_display_sql

def answer_query(question: str, context_rows: pd.DataFrame, variable: str = "temp_c") -> tuple[str, str]:
    """
    Runs the full RAG + LLM call and returns a strictly data-grounded answer.

    If context_rows is empty the LLM is never called — a fixed "no data"
    message is returned instead.  The prompt explicitly forbids hallucination
    and geographic inference; the LLM only summarises the filtered rows.

    Args:
        question: The user's natural language question.
        context_rows: DataFrame of retrieved ARGO rows (already filtered).
        variable: The primary variable requested.

    Returns:
        A tuple (answer_text, sql_string).
    """
    parsed = parse_query(question)
    
    # Early exit: no data → deterministic answer, no LLM call
    if context_rows is None or context_rows.empty:
        return "No data found for the requested region.", to_display_sql(parsed)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    prompt = build_prompt(question, context_rows, variable)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()

    sql = to_display_sql(parsed)
    return answer, sql