"""
structured_query/engine.py

Public entry point for the structured query path.
Delegates entirely to service.answer().

The global _df / load_data() / Parquet loading that lived here
has been removed — all data access now goes through PostgreSQL
via structured_query.repository.
"""
import logging

from structured_query.service import answer as _service_answer

logger = logging.getLogger(__name__)


def answer_structured_query(question: str) -> dict:
    """
    Evaluates a structured natural-language query against PostgreSQL.

    Args:
        question: The user's natural language question.

    Returns:
        dict with keys:
            'summary'  – human-readable result string
            'rows'     – pandas DataFrame of matching rows
            'metadata' – dict describing applied filters and record count
    """
    logger.info("[STRUCTURED] question=%r", question)
    return _service_answer(question)
