"""
structured_query/parser.py

Centralized parsing logic for structured queries.
Extracts depth constraints and geographic region bounds.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from llm.geo_mapping import detect_region
from schema.variables import detect_variable

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    variable: Optional[str] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    date_min: Optional[date] = None
    date_max: Optional[date] = None
    metadata_filters: dict[str, str] = field(default_factory=dict)

    @property
    def has_filters(self) -> bool:
        return any(v is not None for v in (
            self.depth_min, self.depth_max,
            self.lat_min, self.lat_max,
            self.lon_min, self.lon_max,
            self.date_min, self.date_max,
            self.value_min, self.value_max
        ))


def _extract_depth(question: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parses depth constraints from a natural-language question.

    Supports:
      - "between X and Ym" / "between X to Ym"
      - "X-Ym" / "X m to Y m"
      - "below Xm" / "greater than Xm" / "deeper than Xm"
      - "above Xm" / "less than Xm" / "shallower than Xm"
      - "around Xm" / "near Xm" / "at Xm" / bare "Xm" (±50 m tolerance window)
    """
    q = question.lower()
    unit = r"(?:meters?|metres?|m)\b"

    # between X and/to/- Y <unit>  (unit on first number is optional)
    m = re.search(
        rf"between\s+(\d+(?:\.\d+)?)\s*(?:{unit})?\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*{unit}", q
    )
    if m:
        return float(m.group(1)), float(m.group(2))

    # X-Ym / X m to Y m
    m = re.search(
        rf"(\d+(?:\.\d+)?)\s*(?:{unit})?\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*{unit}", q
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

    # around / near / at / bare number X <unit>  →  ±50 m window
    m = re.search(rf"(?:around|near|at\s+)?(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        val = float(m.group(1))
        return max(0.0, val - 50), val + 50

    return None, None


def _extract_date(question: str) -> tuple[Optional[date], Optional[date]]:
    """
    Parses year constraints from a natural language question.
    Only supports unambiguous patterns for now.
    """
    q = question.lower()
    
    # "in {year}" or "during {year}"
    m = re.search(r"\b(?:in|during)\s+(19\d{2}|20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        return date(year, 1, 1), date(year, 12, 31)

    # "since {year}" / "after {year}"
    m = re.search(r"\b(?:since|after)\s+(19\d{2}|20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        return date(year, 1, 1), None

    # "before {year}"
    m = re.search(r"\bbefore\s+(19\d{2}|20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        return None, date(year, 12, 31)

    return None, None


def _extract_value(question: str, variable: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parses generic value constraints (e.g., 'salinity above 35').
    """
    q = question.lower()
    
    # Use negative lookahead to ensure the number is NOT followed by 'm' or 'meters'
    # Also add (?!\.?\d) to prevent the regex engine from backtracking and partially matching a number (e.g. matching "1" from "10m")
    unit_neg = r"(?!\.?\d)(?!\s*(?:m|meters|metres)\b)"
    
    # between X and Y
    m = re.search(rf"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?){unit_neg}", q)
    if m:
        return float(m.group(1)), float(m.group(2))

    # above / greater than X
    m = re.search(rf"\b(?:above|greater than|>)\s+(\d+(?:\.\d+)?){unit_neg}", q)
    if m:
        return float(m.group(1)), None

    # below / less than X
    m = re.search(rf"\b(?:below|less than|<)\s+(\d+(?:\.\d+)?){unit_neg}", q)
    if m:
        return None, float(m.group(1))

    return None, None


def parse_query(question: str) -> ParsedQuery:
    """
    Parses a natural language question into a ParsedQuery containing
    all extracted filters and metadata.
    """
    parsed = ParsedQuery()

    # -- Parse Variable --
    parsed.variable = detect_variable(question)
    if parsed.variable:
        logger.info("[PARSER] Extracted variable: %s", parsed.variable)
        
        # Parse value range if variable exists
        v_min, v_max = _extract_value(question, parsed.variable)
        if v_min is not None or v_max is not None:
            parsed.value_min = v_min
            parsed.value_max = v_max
            lo = f"{v_min}" if v_min is not None else "0"
            hi = f"{v_max}" if v_max is not None else "∞"
            parsed.metadata_filters["value"] = f"{lo}–{hi}"
            logger.info("[PARSER] Extracted value range: %s–%s", lo, hi)

    # -- Parse Depth --
    d_min, d_max = _extract_depth(question)
    if d_min is not None or d_max is not None:
        parsed.depth_min = d_min
        parsed.depth_max = d_max
        lo = f"{d_min:.0f}" if d_min is not None else "0"
        hi = f"{d_max:.0f}" if d_max is not None else "∞"
        parsed.metadata_filters["depth"] = f"{lo}–{hi}m"
        logger.info("[PARSER] Extracted depth: %s–%sm", lo, hi)

    # -- Parse Date --
    date_min, date_max = _extract_date(question)
    if date_min is not None or date_max is not None:
        parsed.date_min = date_min
        parsed.date_max = date_max
        d_lo = date_min.year if date_min else ""
        d_hi = date_max.year if date_max else ""
        if date_min and date_max and date_min.year == date_max.year:
            parsed.metadata_filters["date"] = f"{date_min.year}"
            logger.info("[PARSER] Extracted date: %s", date_min.year)
        else:
            lo_str = f"{d_lo}" if d_lo else "..."
            hi_str = f"{d_hi}" if d_hi else "..."
            parsed.metadata_filters["date"] = f"{lo_str}–{hi_str}"
            logger.info("[PARSER] Extracted date: %s–%s", lo_str, hi_str)

    # -- Parse Region --
    name, bounds = detect_region(question)
    if name and bounds:
        parsed.lat_min = bounds["lat_min"]
        parsed.lat_max = bounds["lat_max"]
        parsed.lon_min = bounds["lon_min"]
        parsed.lon_max = bounds["lon_max"]
        parsed.metadata_filters["region"] = name.title()
        logger.info("[PARSER] Extracted region: %s", name.title())
    else:
        # Detect any generic ocean/sea words for context even if unmatched
        q_lower = question.lower()
        if "ocean" in q_lower or "sea" in q_lower:
            # We didn't match a specific bounding box
            parsed.metadata_filters["unmatched_region"] = True

    return parsed


def to_display_sql(parsed: ParsedQuery) -> str:
    """
    Generates a deterministic, display-only SQL string representing
    the parsed constraints. This replaces the slow/unreliable LLM SQL generation.
    """
    conditions = []
    
    if parsed.lat_min is not None and parsed.lat_max is not None:
        conditions.append(f"latitude BETWEEN {parsed.lat_min} AND {parsed.lat_max}")
    if parsed.lon_min is not None and parsed.lon_max is not None:
        conditions.append(f"longitude BETWEEN {parsed.lon_min} AND {parsed.lon_max}")
    if parsed.depth_min is not None:
        conditions.append(f"depth_m >= {parsed.depth_min}")
    if parsed.depth_max is not None:
        conditions.append(f"depth_m <= {parsed.depth_max}")
    if parsed.date_min is not None:
        conditions.append(f"date >= '{parsed.date_min.isoformat()}'")
    if parsed.date_max is not None:
        conditions.append(f"date <= '{parsed.date_max.isoformat()}'")
    
    variable = parsed.variable if parsed.variable else "temp_c"
    
    if parsed.value_min is not None:
        conditions.append(f"{variable} >= {parsed.value_min}")
    if parsed.value_max is not None:
        conditions.append(f"{variable} <= {parsed.value_max}")
    
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else "WHERE 1=1"
    
    return f"SELECT float_id, date, latitude, longitude, depth_m, {variable}\nFROM argo_profiles\n{where_clause}"
