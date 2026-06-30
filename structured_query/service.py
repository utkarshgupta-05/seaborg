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


_STAT_KEY = {
    "temp_c": ("temp", "°C", "Temperature"),
    "salinity": ("salinity", " PSU", "Salinity"),
    "oxygen": ("oxygen", "", "Oxygen"),
    "chlorophyll": ("chlorophyll", "", "Chlorophyll"),
    "nitrate": ("nitrate", "", "Nitrate"),
}

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
    requested_variable = parsed.variable or "temp_c"

    # -- Guard: no filters → skip row fetching but fetch global aggregate stats
    if not parsed.has_filters:
        logger.info("[STRUCTURED] No filters found. Performing global aggregation query.")
        
        stats = repository.aggregate_stats_for_variable(requested_variable)
            
        if not stats or stats.get("count", 0) == 0:
            return _empty_response(parsed.metadata_filters)

        summary = _build_summary(stats.get("count", 0), stats, requested_variable)
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
        date_min=parsed.date_min,
        date_max=parsed.date_max,
        variable=requested_variable,
        value_min=parsed.value_min,
        value_max=parsed.value_max,
    )

    count = len(rows_df)
    logger.info("[STRUCTURED] Rows returned: %d", count)

    if rows_df.empty:
        return _empty_response(parsed.metadata_filters)

    stats = repository.aggregate_stats_for_variable(
        requested_variable,
        depth_min=parsed.depth_min,
        depth_max=parsed.depth_max,
        lat_min=parsed.lat_min,
        lat_max=parsed.lat_max,
        lon_min=parsed.lon_min,
        lon_max=parsed.lon_max,
        date_min=parsed.date_min,
        date_max=parsed.date_max,
        value_min=parsed.value_min,
        value_max=parsed.value_max,
    )

    summary = _build_summary(count, stats, requested_variable)

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


def _build_summary(count: int, stats: dict, variable: str) -> str:
    parts = [f"Found {count:,} matching observations."]

    # Unconditional Depth range (if requested variable is NOT depth_m)
    if variable != "depth_m":
        if stats.get("min_depth") is not None and stats.get("max_depth") is not None:
            parts.append(
                f"\nDepth range:\n{stats['min_depth']:.0f}m–{stats['max_depth']:.0f}m"
            )

    # Lead with requested variable's stats
    if variable == "depth_m":
        if stats.get("min_depth") is not None and stats.get("max_depth") is not None:
            parts.append(
                f"\nDepth range:\n{stats['min_depth']:.0f}m–{stats['max_depth']:.0f}m"
            )
            if stats.get("avg_depth") is not None:
                parts.append(f"\nAverage depth:\n{stats['avg_depth']:.1f}m")
        else:
            parts.append(f"\nNo Depth measurements were found among the {count} matching observations.")
    elif variable in _STAT_KEY:
        key, unit, title = _STAT_KEY[variable]
        if stats.get(f"avg_{key}") is not None:
            parts.append(
                f"\n{title} range:\n{stats[f'min_{key}']:.2f}{unit}–{stats[f'max_{key}']:.2f}{unit}"
                f"\n\nAverage {title.lower()}:\n{stats[f'avg_{key}']:.2f}{unit}"
            )
        else:
            parts.append(f"\nNo {title} measurements were found among the {count} matching observations.")

    # Add other variable stats as secondary context (only if they are NOT the requested variable)
    for v_key, (key, unit, title) in _STAT_KEY.items():
        if variable != v_key and stats.get(f"avg_{key}") is not None:
            parts.append(f"\nAverage {title.lower()}:\n{stats[f'avg_{key}']:.2f}{unit}")

    return "\n".join(parts)
