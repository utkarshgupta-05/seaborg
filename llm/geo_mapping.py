"""
Geographic region mapping for ARGO ocean/sea queries.

Maps human-readable region names (oceans, seas) to bounding-box
latitude/longitude ranges so the NL → SQL pipeline can inject
correct spatial filters.
"""

# ── Region definitions ────────────────────────────────────────────
# Each entry: { "lat_min", "lat_max", "lon_min", "lon_max" }

REGION_BOUNDS: dict[str, dict[str, float]] = {
    # Oceans
    "indian ocean":      {"lat_min": -30, "lat_max":  30, "lon_min":   20, "lon_max": 120},
    "pacific ocean":     {"lat_min": -30, "lat_max":  30, "lon_min":  120, "lon_max": 290},
    "atlantic ocean":    {"lat_min": -30, "lat_max":  30, "lon_min":  -70, "lon_max":  20},
    "southern ocean":    {"lat_min": -90, "lat_max": -60, "lon_min": -180, "lon_max": 180},
    "arctic ocean":      {"lat_min":  60, "lat_max":  90, "lon_min": -180, "lon_max": 180},
    # Seas
    "arabian sea":       {"lat_min":   5, "lat_max":  25, "lon_min":   50, "lon_max":  75},
    "bay of bengal":     {"lat_min":   5, "lat_max":  25, "lon_min":   80, "lon_max": 100},
    "south china sea":   {"lat_min":   0, "lat_max":  25, "lon_min":  100, "lon_max": 120},
    "mediterranean sea": {"lat_min":  30, "lat_max":  45, "lon_min":   -5, "lon_max":  35},
    "gulf of mexico":    {"lat_min":  18, "lat_max":  30, "lon_min":  -98, "lon_max": -81},
    "caribbean sea":     {"lat_min":   9, "lat_max":  22, "lon_min":  -89, "lon_max": -60},
    "red sea":           {"lat_min":  12, "lat_max":  30, "lon_min":   32, "lon_max":  43},
    "black sea":         {"lat_min":  40, "lat_max":  47, "lon_min":   27, "lon_max":  42},
    "baltic sea":        {"lat_min":  53, "lat_max":  66, "lon_min":   10, "lon_max":  30},
    "north sea":         {"lat_min":  51, "lat_max":  61, "lon_min":   -4, "lon_max":  10},
}


def map_region_to_coordinates(region_name: str) -> dict | None:
    """
    Look up bounding-box coordinates for a named ocean or sea.

    Args:
        region_name: Free-text region name (case-insensitive).

    Returns:
        Dict with keys lat_min, lat_max, lon_min, lon_max if the region
        is recognised, otherwise None.
    """
    return REGION_BOUNDS.get(region_name.strip().lower())


def detect_region(text: str) -> tuple[str | None, dict | None]:
    """
    Scan *text* for the first known region name (longest-match-first).

    Returns:
        (matched_name, bounds_dict)  — or (None, None) if no region found.
    """
    text_lower = text.lower()
    # Sort by key length descending so "south china sea" matches before
    # a hypothetical shorter overlap.
    for name in sorted(REGION_BOUNDS, key=len, reverse=True):
        if name in text_lower:
            return name, REGION_BOUNDS[name]
    return None, None


def detect_region_from_coords(lat: float, lon: float) -> str | None:
    """
    Reverse-geocodes coordinates into a known region name.

    Returns:
        The matched region name, or None if the coordinates don't fall
        into any known bounding box.
    """
    for name, bounds in REGION_BOUNDS.items():
        if (bounds["lat_min"] <= lat <= bounds["lat_max"] and
            bounds["lon_min"] <= lon <= bounds["lon_max"]):
            return name.title()
    return None
