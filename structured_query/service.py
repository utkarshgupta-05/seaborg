"""
structured_query/service.py

Orchestration layer for structured queries.
Parses the natural language question, delegates to the repository,
and assembles the final response dict used by chat.py.
"""
import logging
from typing import Optional

import pandas as pd

from structured_query import repository
from structured_query.parser import parse_query

logger = logging.getLogger(__name__)


# ── Service entry point ───────────────────────────────────────────────────────

def answer(question: str) -> dict:
    """
    Parses the question, queries PostgreSQL, and returns the response dict.

    Returns:
        {
            "summary": str,
            "rows":    pd.DataFrame,
            "metadata": dict,
        }

    This is the same shape that the old engine.answer_structured_query()
    returned, so chat.py needs no changes.
    """
    # -- Parse the query using the centralized parser --------------------------
    parsed = parse_query(question)

    # -- Guard: no filters → skip row fetching but fetch global aggregate stats
    if not parsed.has_filters:
        logger.info("[STRUCTURED] No filters found. Performing global aggregation query.")
        stats = repository.aggregate_stats()
        if not stats or stats.get("count", 0) == 0:
            return _empty_response(parsed.metadata_filters)

        summary = _build_summary(stats.get("count", 0), stats)
        return {
            "summary": summary,
            "rows": pd.DataFrame(),
            "metadata": {
                "query_type": "structured",
                "filters": parsed.metadata_filters,
                "record_count": stats.get("count", 0),
            },
        }

    # -- Fetch rows via repository --------------------------------------------
    rows_df = repository.query_with_filters(
        depth_min=parsed.depth_min,
        depth_max=parsed.depth_max,
        lat_min=parsed.lat_min,
        lat_max=parsed.lat_max,
        lon_min=parsed.lon_min,
        lon_max=parsed.lon_max,
    )

    count = len(rows_df)
    logger.info("[STRUCTURED] Rows returned: %d", count)

    if rows_df.empty:
        return _empty_response(parsed.metadata_filters)

    # -- Fetch aggregate stats ------------------------------------------------
    stats = repository.aggregate_stats(
        depth_min=parsed.depth_min,
        depth_max=parsed.depth_max,
        lat_min=parsed.lat_min,
        lat_max=parsed.lat_max,
        lon_min=parsed.lon_min,
        lon_max=parsed.lon_max,
    )

    summary = _build_summary(count, stats)

    return {
        "summary": summary,
        "rows": rows_df,
        "metadata": {
            "query_type": "structured",
            "filters": parsed.metadata_filters,
            "record_count": count,
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_response(metadata_filters: dict) -> dict:
    return {
        "summary": "No matching observations found for the given filters.",
        "rows": pd.DataFrame(),
        "metadata": {
            "query_type": "structured",
            "filters": metadata_filters,
            "record_count": 0,
        },
    }


def _build_summary(count: int, stats: dict) -> str:
    parts = [f"Found {count:,} matching observations."]

    if stats.get("min_depth") is not None and stats.get("max_depth") is not None:
        parts.append(
            f"\nDepth range:\n{stats['min_depth']:.0f}m–{stats['max_depth']:.0f}m"
        )

    if stats.get("avg_temp") is not None:
        parts.append(
            f"\nTemperature range:\n{stats['min_temp']:.2f}°C–{stats['max_temp']:.2f}°C"
            f"\n\nAverage temperature:\n{stats['avg_temp']:.2f}°C"
        )
    else:
        parts.append("\nTemperature range:\nN/A\n\nAverage temperature:\nN/A")

    if stats.get("avg_salinity") is not None:
        parts.append(f"\nAverage salinity:\n{stats['avg_salinity']:.2f} PSU")

    return "\n".join(parts)
