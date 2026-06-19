"""
structured_query/service.py

Orchestration layer for structured queries.
Parses the natural language question, delegates to the repository,
and assembles the final response dict used by chat.py.
"""
import logging
import re
from typing import Optional

import pandas as pd

from llm.geo_mapping import detect_region
from structured_query import repository

logger = logging.getLogger(__name__)


# ── Depth extraction ──────────────────────────────────────────────────────────

def extract_depth(question: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parses depth constraints from a natural-language question.

    Supports:
      - "between X and Ym" / "between X to Ym"
      - "below Xm" / "greater than Xm" / "deeper than Xm"
      - "above Xm" / "less than Xm" / "shallower than Xm"
      - "at Xm" / bare "Xm"  (±50 m tolerance window)

    Also accepts "meters" / "metre" / "m" as the unit.

    Returns:
        (depth_min, depth_max) — either value may be None.
    """
    q = question.lower()
    unit = r"(?:meters?|metres?|m)\b"

    # between X and/to/- Y <unit>  (unit on first number is optional)
    m = re.search(
        rf"between\s+(\d+(?:\.\d+)?)\s*(?:{unit})?\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*{unit}", q
    )
    if m:
        return float(m.group(1)), float(m.group(2))

    # below / greater than / deeper than X <unit>
    m = re.search(rf"(?:below|greater than|deeper than)\s+(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        return float(m.group(1)), None

    # above / less than / shallower than X <unit>
    m = re.search(rf"(?:above|less than|shallower than)\s+(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        return None, float(m.group(1))

    # around / at / bare number X <unit>  →  ±50 m window
    m = re.search(rf"(?:around|at\s+)?(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        val = float(m.group(1))
        return max(0.0, val - 50), val + 50

    return None, None


# ── Region extraction ─────────────────────────────────────────────────────────

def extract_region(question: str) -> Optional[dict]:
    """
    Returns a bounds dict (with 'name' key added) if a known region is found,
    otherwise None.
    """
    name, bounds = detect_region(question)
    if name and bounds:
        return {**bounds, "name": name}
    return None


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
    metadata_filters: dict = {}

    # -- Parse depth -----------------------------------------------------------
    depth_min, depth_max = extract_depth(question)

    if depth_min is not None or depth_max is not None:
        lo = f"{depth_min:.0f}" if depth_min is not None else "0"
        hi = f"{depth_max:.0f}" if depth_max is not None else "∞"
        metadata_filters["depth"] = f"{lo}–{hi}m"
        logger.info("[STRUCTURED] Depth filter: %s–%sm", lo, hi)

    # -- Parse region ----------------------------------------------------------
    lat_min = lat_max = lon_min = lon_max = None
    bounds = extract_region(question)
    if bounds:
        region_name = bounds.pop("name", "Unknown Region")
        lat_min = bounds["lat_min"]
        lat_max = bounds["lat_max"]
        lon_min = bounds["lon_min"]
        lon_max = bounds["lon_max"]
        metadata_filters["region"] = region_name.title()
        logger.info("[STRUCTURED] Region filter: %s", region_name)

    # -- Guard: no filters → skip row fetching but fetch global aggregate stats
    has_filter = any(
        v is not None
        for v in (depth_min, depth_max, lat_min, lat_max, lon_min, lon_max)
    )
    if not has_filter:
        logger.info("[STRUCTURED] No filters found. Performing global aggregation query.")
        stats = repository.aggregate_stats()
        if not stats or stats.get("count", 0) == 0:
            return _empty_response(metadata_filters)

        summary = _build_summary(stats.get("count", 0), stats)
        return {
            "summary": summary,
            "rows": pd.DataFrame(),
            "metadata": {
                "query_type": "structured",
                "filters": metadata_filters,
                "record_count": stats.get("count", 0),
            },
        }

    # -- Fetch rows via repository --------------------------------------------
    rows_df = repository.query_with_filters(
        depth_min=depth_min,
        depth_max=depth_max,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    count = len(rows_df)
    logger.info("[STRUCTURED] Rows returned: %d", count)

    if rows_df.empty:
        return _empty_response(metadata_filters)

    # -- Fetch aggregate stats ------------------------------------------------
    stats = repository.aggregate_stats(
        depth_min=depth_min,
        depth_max=depth_max,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    summary = _build_summary(count, stats)

    return {
        "summary": summary,
        "rows": rows_df,
        "metadata": {
            "query_type": "structured",
            "filters": metadata_filters,
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
