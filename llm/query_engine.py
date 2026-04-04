import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .prompts import build_prompt
from .nl_to_sql import generate_sql

load_dotenv()


def answer_query(question: str, context_rows: pd.DataFrame) -> tuple[str, str]:
    """
    Runs the full RAG + LLM call and returns a grounded answer with SQL.

    Args:
        question: The user's natural language question.
        context_rows: DataFrame of retrieved ARGO rows from retriever.retrieve().

    Returns:
        A tuple (answer_text, sql_string) where answer_text is the LLM's
        response grounded in context_rows, and sql_string is the generated
        SQL for the question.

    Side effects:
        Makes two OpenAI API calls — one for the answer, one for SQL generation.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    prompt = build_prompt(question, context_rows)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()

    try:
        sql = generate_sql(question)
    except Exception:
        sql = "-- SQL generation failed"

    return answer, sql