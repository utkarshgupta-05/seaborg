import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from .prompts import build_prompt
from .nl_to_sql import generate_sql

load_dotenv()


def answer_query(question: str, context_rows: pd.DataFrame) -> tuple[str, str]:
    """
    Runs the full RAG + LLM call and returns a strictly data-grounded answer.

    If context_rows is empty the LLM is never called — a fixed "no data"
    message is returned instead.  The prompt explicitly forbids hallucination
    and geographic inference; the LLM only summarises the filtered rows.

    Args:
        question: The user's natural language question.
        context_rows: DataFrame of retrieved ARGO rows (already filtered).

    Returns:
        A tuple (answer_text, sql_string).

    Side effects:
        Makes up to two Groq API calls (answer + SQL generation).
    """
    # Generate SQL regardless — the caller may still want it
    try:
        sql = generate_sql(question)
    except Exception:
        sql = "-- SQL generation failed"

    # Early exit: no data → deterministic answer, no LLM call
    if context_rows is None or context_rows.empty:
        return "No data found for the requested region.", sql

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    prompt = build_prompt(question, context_rows)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()

    return answer, sql