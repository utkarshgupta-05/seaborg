"""
structured_query/parser.py

Centralized parsing logic for structured queries.
Extracts depth constraints and geographic region bounds.
"""
import logging
import re
from dataclasses import dataclass, field
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
    metadata_filters: dict[str, str] = field(default_factory=dict)

    @property
    def has_filters(self) -> bool:
        return any(v is not None for v in (
            self.depth_min, self.depth_max,
            self.lat_min, self.lat_max,
            self.lon_min, self.lon_max
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

    # -- Parse Depth --
    d_min, d_max = _extract_depth(question)
    if d_min is not None or d_max is not None:
        parsed.depth_min = d_min
        parsed.depth_max = d_max
        lo = f"{d_min:.0f}" if d_min is not None else "0"
        hi = f"{d_max:.0f}" if d_max is not None else "∞"
        parsed.metadata_filters["depth"] = f"{lo}–{hi}m"
        logger.info("[PARSER] Extracted depth: %s–%sm", lo, hi)

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
