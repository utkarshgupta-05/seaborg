import os

import faiss
import pandas as pd
from dotenv import load_dotenv

from .embedder import embed_query
from structured_query.parser import ParsedQuery


_index = None
_df = None

_ARGO_SCHEMA_COLUMNS = [
    "float_id",
    "date",
    "latitude",
    "longitude",
    "depth_m",
    "temp_c",
    "salinity",
    "oxygen",
    "chlorophyll",
    "nitrate",
    "faiss_distance",
]


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures returned DataFrame matches argo_profiles column schema/order."""
    out = df.copy()
    for col in _ARGO_SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[_ARGO_SCHEMA_COLUMNS]


def load_index() -> None:
    """
    Loads FAISS index and parquet DataFrame once into module-level state.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Reads files from paths configured in environment variables.
    """
    global _index, _df
    if _index is not None and _df is not None:
        return

    load_dotenv()
    faiss_index_path = os.getenv("FAISS_INDEX_PATH")
    parquet_path = os.getenv("PARQUET_PATH")

    if not faiss_index_path:
        raise ValueError("FAISS_INDEX_PATH is not set.")
    if not parquet_path:
        raise ValueError("PARQUET_PATH is not set.")

    _index = faiss.read_index(faiss_index_path)
    _df = pd.read_parquet(parquet_path).reset_index(drop=True)


def retrieve(user_query: str, top_k: int = 5, distance_threshold: float = None, parsed_query: ParsedQuery = None, variable: str = "temp_c") -> pd.DataFrame:
    """
    Retrieves top-k nearest rows from parquet using FAISS similarity search.

    Args:
        user_query: Natural language user query.
        top_k: Number of rows to return.
        distance_threshold: Optional max L2 distance. Rows with distance > threshold are dropped.
        parsed_query: Optional ParsedQuery object containing depth and geographic filters.

    Returns:
        DataFrame of retrieved rows with argo_profiles schema columns (plus faiss_distance).

    Side effects:
        None.
    """
    if _index is None or _df is None:
        raise RuntimeError("Index not loaded. Call load_index() before retrieve().")

    query_vec = embed_query(user_query).astype("float32")
    
    # Pre-filtering tradeoff: Rebuilding FAISS indices per query or using IDMaps adds significant
    # complexity and latency. Instead, we use an efficient post-retrieval filter by fetching a
    # larger initial set (top 2000) and applying strict Pandas filtering before truncating to top_k.
    _SPARSE_VARIABLES = {"oxygen", "chlorophyll", "nitrate"}
    search_k = len(_df) if variable in _SPARSE_VARIABLES else (2000 if parsed_query and parsed_query.has_filters else top_k)
    search_k = min(search_k, len(_df))
    distances, indices = _index.search(query_vec, search_k)
    
    rows = _df.iloc[indices[0]].copy().reset_index(drop=True)
    rows["faiss_distance"] = distances[0]
    
    if distance_threshold is not None:
        rows = rows[rows["faiss_distance"] <= distance_threshold].reset_index(drop=True)

    # Apply strict constraints if parsed
    if parsed_query:
        if parsed_query.lat_min is not None:
            rows = rows[(rows["latitude"] >= parsed_query.lat_min) & (rows["latitude"] <= parsed_query.lat_max)]
        if parsed_query.lon_min is not None:
            rows = rows[(rows["longitude"] >= parsed_query.lon_min) & (rows["longitude"] <= parsed_query.lon_max)]
        if parsed_query.depth_min is not None:
            rows = rows[rows["depth_m"] >= parsed_query.depth_min]
        if parsed_query.depth_max is not None:
            rows = rows[rows["depth_m"] <= parsed_query.depth_max]
        if parsed_query.date_min is not None:
            rows = rows[pd.to_datetime(rows["date"]) >= pd.to_datetime(parsed_query.date_min)]
        if parsed_query.date_max is not None:
            rows = rows[pd.to_datetime(rows["date"]) <= pd.to_datetime(parsed_query.date_max)]

    _SPARSE_VARIABLES = {"oxygen", "chlorophyll", "nitrate"}
    if variable in _SPARSE_VARIABLES:
        rows["_is_null"] = rows[variable].isna().astype(int)
        rows = rows.sort_values(["_is_null", "faiss_distance"]).drop(columns=["_is_null"])

    rows = rows.head(top_k).reset_index(drop=True)
    return _ensure_schema(rows)
