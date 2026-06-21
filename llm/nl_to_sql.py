import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


from .prompts import SQL_PROMPT
from .geo_mapping import detect_region

load_dotenv()




def _preprocess_question(question: str) -> str:
    """
    Detect a geographic region in the question and append coordinate
    context so the LLM can generate correct lat/lon WHERE clauses.
    """
    region_name, bounds = detect_region(question)
    if bounds is None:
        return question
    hint = (
        f" (Note: '{region_name}' corresponds to latitude BETWEEN "
        f"{bounds['lat_min']} AND {bounds['lat_max']} AND longitude "
        f"BETWEEN {bounds['lon_min']} AND {bounds['lon_max']})"
    )
    return question + hint


def generate_sql(question: str) -> str:
    """
    Sends SQL_PROMPT to the LLM and returns a raw SQL string.

    If the question mentions a known ocean or sea, coordinate context is
    injected automatically so the LLM generates correct spatial filters.

    Args:
        question: The user's natural language question.

    Returns:
        A raw SQL string from the LLM. May be unsafe — this is only for
        display and should not be executed directly.

    Side effects:
        Makes a Groq API call.
    """
    enriched = _preprocess_question(question)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": SQL_PROMPT.format(question=enriched),
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()
