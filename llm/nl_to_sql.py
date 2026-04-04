import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.engine import Engine
from sqlalchemy import text

from .prompts import SQL_PROMPT

load_dotenv()

FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT"
]


def generate_sql(question: str) -> str:
    """
    Sends SQL_PROMPT to the LLM and returns a raw SQL string.

    Args:
        question: The user's natural language question.

    Returns:
        A raw SQL string from the LLM. May be unsafe — always validate
        with safe_sql_query() before executing.

    Side effects:
        Makes an OpenAI API call.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": SQL_PROMPT.format(question=question),
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def safe_sql_query(
    sql: str, engine: Engine
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Validates sql against forbidden keywords and executes if safe.

    Args:
        sql: SQL string to validate and execute.
        engine: SQLAlchemy engine connected to the seaborg database.

    Returns:
        (DataFrame, None) if the query is safe and executes successfully.
        (None, error_message) if the query contains forbidden keywords or fails.

    Side effects:
        Executes a SELECT query against PostgreSQL if safe.
    """
    sql_upper = sql.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in sql_upper:
            return None, f"Unsafe SQL rejected — forbidden keyword detected: {keyword}"

    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(sql), conn)
        return result, None
    except Exception as exc:
        return None, f"SQL execution error: {str(exc)}"