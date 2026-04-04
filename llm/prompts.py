import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CHAT_PROMPT = """You are SeaBorg, an expert ocean data analyst. Answer the user's question using
ONLY the data records provided below. Be specific, cite float IDs and dates.
If the data does not support the question, say so clearly.

Context records:
{context}

Question: {question}

Answer:"""

SQL_PROMPT = """Convert the following question into a valid PostgreSQL SELECT query for the
table `argo_profiles` with columns:
id, float_id, date, latitude, longitude, depth_m, temp_c, salinity, oxygen, created_at.

Return ONLY the SQL query. No explanation. No markdown. No semicolon at the end.

Question: {question}"""


def build_prompt(question: str, context_rows: pd.DataFrame) -> str:
    """
    Formats context_rows as a bullet list and fills CHAT_PROMPT.

    Args:
        question: The user's natural language question.
        context_rows: DataFrame of retrieved ARGO rows from retriever.retrieve().

    Returns:
        A fully formatted prompt string ready to send to the LLM.

    Side effects:
        None.
    """
    bullets = []
    for _, row in context_rows.iterrows():
        bullets.append(
            f"• Float {row['float_id']} | {row['date']} | "
            f"{row['depth_m']:.0f}m | {row['temp_c']:.1f}°C | "
            f"{row['salinity']:.2f} PSU"
        )
    context = "\n".join(bullets) if bullets else "No records retrieved."
    return CHAT_PROMPT.format(context=context, question=question)