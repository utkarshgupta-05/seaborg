import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CHAT_PROMPT = """You are SeaBorg, a strict ocean data analyst.

RULES — follow every one without exception:
1. Answer ONLY using the provided data records below. Never use outside knowledge.
2. Do NOT infer or assume geographic location from coordinates. The data has
   already been filtered to the correct region — just summarise what you see.
3. Do NOT hallucinate or assume missing information. If information is not
   present in the records, say it is not available.
4. Do NOT claim a float belongs to a specific ocean or sea unless the question
   already names that region. Even then, do not validate or contradict — just
   report the numbers.
5. Focus on concrete values: cite float IDs, dates, temperature ranges,
   averages, and trends visible in the data.
6. If the records are empty or say "No records retrieved", respond exactly:
   "No data found for the requested region."

Data records (already filtered for the user's query):
{context}

Question: {question}

Answer:"""

SQL_PROMPT = """Convert the following question into a valid PostgreSQL SELECT query for the
table `argo_profiles` with columns:
id, float_id, date, latitude, longitude, depth_m, temp_c, salinity, oxygen, created_at.

IMPORTANT: The table does NOT have a region or ocean column. When a question
references a named ocean or sea, you MUST filter by latitude and longitude
ranges. If coordinate hints are provided in parentheses at the end of the
question, use those exact BETWEEN values.

Return ONLY the SQL query. No explanation. No markdown. No semicolon at the end.

Question: {question}"""


def build_prompt(question: str, context_rows: pd.DataFrame) -> str:
    """
    Formats context_rows as a bullet list and fills CHAT_PROMPT.

    Only the filtered data rows are included — no geographic context is
    injected so the LLM cannot hallucinate region assignments.  Rows are
    capped at 10 to keep the prompt focused.

    Args:
        question: The user's natural language question.
        context_rows: DataFrame of retrieved ARGO rows from retriever.retrieve().

    Returns:
        A fully formatted prompt string ready to send to the LLM.

    Side effects:
        None.
    """
    # Limit to 10 rows to keep the prompt concise and focused
    limited = context_rows.head(10)

    bullets = []
    for _, row in limited.iterrows():
        bullets.append(
            f"• Float {row['float_id']} | {row['date']} | "
            f"{row['depth_m']:.0f}m | {row['temp_c']:.1f}°C | "
            f"{row['salinity']:.2f} PSU"
        )
    context = "\n".join(bullets) if bullets else "No records retrieved."

    return CHAT_PROMPT.format(context=context, question=question)