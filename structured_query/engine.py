import logging
import os
import re

import pandas as pd
from dotenv import load_dotenv

from llm.geo_mapping import detect_region

logger = logging.getLogger(__name__)

_df: pd.DataFrame | None = None


def load_data() -> None:
    """Loads Parquet DataFrame once into module-level state."""
    global _df
    if _df is not None:
        return

    load_dotenv()
    parquet_path = os.getenv("PARQUET_PATH", "data/processed/argo.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    _df = pd.read_parquet(parquet_path).reset_index(drop=True)


def extract_depth(question: str) -> tuple[float | None, float | None]:
    """
    Extracts depth constraints from question.
    Returns (depth_min, depth_max).
    """
    q = question.lower()
    
    # Check for "between X and Ym" or "between X to Ym"
    match = re.search(r"between\s+(\d+)\s*(?:and|to|-)\s*(\d+)\s*m\b", q)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Check for "below Xm" or "greater than Xm"
    match = re.search(r"(?:below|greater than)\s+(\d+)\s*m\b", q)
    if match:
        return float(match.group(1)), None
        
    # Check for "above Xm" or "less than Xm"
    match = re.search(r"(?:above|less than)\s+(\d+)\s*m\b", q)
    if match:
        return None, float(match.group(1))
        
    # Check for "at Xm" or just "Xm"
    match = re.search(r"(?:at\s+)?(\d+)\s*m\b", q)
    if match:
        val = float(match.group(1))
        return val - 50, val + 50
        
    return None, None


def extract_region(question: str) -> dict | None:
    """Returns geographical bounds dict if region found in question."""
    name, bounds = detect_region(question)
    if name:
        bounds["name"] = name  # Include name for logging
    return bounds


def answer_structured_query(question: str) -> dict:
    """
    Evaluates structured query on the dataframe deterministically.
    Returns dict: {'summary': str, 'rows': pd.DataFrame, 'metadata': dict}
    """
    load_data()
    global _df
    
    print("[ROUTER] STRUCTURED")
    
    # Create mask
    mask = pd.Series(True, index=_df.index)
    
    # Track metadata filters
    metadata_filters = {}
    
    # Region filter
    bounds = extract_region(question)
    if bounds:
        region_name = bounds.pop("name", "Unknown Region")
        print(f"[FILTER]\nRegion: {region_name.title()}")
        metadata_filters["region"] = region_name.title()
        mask &= (_df["latitude"] >= bounds["lat_min"]) & (_df["latitude"] <= bounds["lat_max"])
        mask &= (_df["longitude"] >= bounds["lon_min"]) & (_df["longitude"] <= bounds["lon_max"])
        
    # Depth filter
    depth_min, depth_max = extract_depth(question)
    if depth_min is not None or depth_max is not None:
        min_str = f"{depth_min:.0f}" if depth_min is not None else "0"
        max_str = f"{depth_max:.0f}" if depth_max is not None else "∞"
        depth_label = f"{min_str}-{max_str}m"
        print(f"[FILTER]\nDepth: {depth_label}")
        metadata_filters["depth"] = depth_label
        if depth_min is not None:
            mask &= (_df["depth_m"] >= depth_min)
        if depth_max is not None:
            mask &= (_df["depth_m"] <= depth_max)

    # Apply mask
    filtered_df = _df[mask].copy()
    count = len(filtered_df)
    print(f"[RESULT]\nRows matched: {count}")
    
    if filtered_df.empty:
        summary = "Found 0 matching observations."
        return {
            "summary": summary,
            "rows": filtered_df,
            "metadata": {
                "query_type": "structured",
                "filters": metadata_filters,
                "record_count": count
            }
        }

    # Generate Statistical Summary
    min_depth = filtered_df["depth_m"].min()
    max_depth = filtered_df["depth_m"].max()
    
    if "temp_c" in filtered_df.columns and not filtered_df["temp_c"].dropna().empty:
        avg_temp = filtered_df["temp_c"].mean()
        min_temp = filtered_df["temp_c"].min()
        max_temp = filtered_df["temp_c"].max()
        temp_str = f"Temperature range:\n{min_temp:.2f}°C–{max_temp:.2f}°C\n\nAverage temperature:\n{avg_temp:.2f}°C"
    else:
        temp_str = "Temperature range:\nN/A\n\nAverage temperature:\nN/A"

    summary = (
        f"Found {count} matching observations.\n\n"
        f"Depth range:\n{min_depth:.0f}m–{max_depth:.0f}m\n\n"
        f"{temp_str}"
    )

    return {
        "summary": summary,
        "rows": filtered_df,
        "metadata": {
            "query_type": "structured",
            "filters": metadata_filters,
            "record_count": count
        }
    }
