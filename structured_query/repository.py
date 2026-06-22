"""
structured_query/repository.py

Pure data-access layer for structured ARGO queries.
All queries run against PostgreSQL via SQLAlchemy.
No Parquet, no Pandas DataFrame loading from files.
"""
import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from api.database import get_engine

load_dotenv()

logger = logging.getLogger(__name__)

from schema.variables import VARIABLE_REGISTRY

_availability_cache = {}

def is_variable_available(variable: str) -> bool:
    """
    Checks if a given variable has any non-null data in the database.
    Caches the result per process. (Note: restart uvicorn to clear cache after ingestion).
    """
    if variable not in VARIABLE_REGISTRY:
        return False
        
    if variable in _availability_cache:
        return _availability_cache[variable]
        
    sql = f"SELECT 1 FROM argo_profiles WHERE {variable} IS NOT NULL LIMIT 1"
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql)).fetchone()
        
    is_avail = result is not None
    _availability_cache[variable] = is_avail
    return is_avail


def query_with_filters(
    *,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Returns rows from argo_profiles matching the given filters.

    All filter arguments are optional. When none are supplied the function
    returns an empty DataFrame — it never does an unguarded full-table scan.

    Args:
        depth_min:  Lower bound for depth_m (inclusive).
        depth_max:  Upper bound for depth_m (inclusive).
        lat_min:    Lower bound for latitude (inclusive).
        lat_max:    Upper bound for latitude (inclusive).
        lon_min:    Lower bound for longitude (inclusive).
        lon_max:    Upper bound for longitude (inclusive).
        limit:      Maximum number of rows to return. Defaults to 500.

    Returns:
        pandas DataFrame with argo_profiles columns, possibly empty.
    """
    # Guard: require at least one filter to avoid accidental full scans
    has_filter = any(
        v is not None
        for v in (depth_min, depth_max, lat_min, lat_max, lon_min, lon_max)
    )
    if not has_filter:
        logger.warning("query_with_filters called with no filters; returning empty.")
        return pd.DataFrame()

    conditions = []
    params: dict = {"limit": limit}

    if depth_min is not None:
        conditions.append("depth_m >= :depth_min")
        params["depth_min"] = depth_min
    if depth_max is not None:
        conditions.append("depth_m <= :depth_max")
        params["depth_max"] = depth_max
    if lat_min is not None:
        conditions.append("latitude >= :lat_min")
        params["lat_min"] = lat_min
    if lat_max is not None:
        conditions.append("latitude <= :lat_max")
        params["lat_max"] = lat_max
    if lon_min is not None:
        conditions.append("longitude >= :lon_min")
        params["lon_min"] = lon_min
    if lon_max is not None:
        conditions.append("longitude <= :lon_max")
        params["lon_max"] = lon_max

    where = " AND ".join(conditions)
    sql = (
        f"SELECT float_id, date, latitude, longitude, depth_m, temp_c, salinity, oxygen "
        f"FROM argo_profiles "
        f"WHERE {where} "
        f"ORDER BY date DESC, depth_m "
        f"LIMIT :limit"
    )

    logger.info("Structured SQL: %s | params: %s", sql.strip(), params)

    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def aggregate_stats(
    *,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
) -> dict:
    """
    Returns aggregate statistics for rows matching the given filters.

    Unlike query_with_filters, this function allows global aggregation
    when no filters are provided — PostgreSQL handles COUNT/AVG/MIN/MAX
    efficiently even on the full table.

    Args:
        depth_min, depth_max, lat_min, lat_max, lon_min, lon_max: same as
            query_with_filters.

    Returns:
        dict with keys: count, avg_temp, min_temp, max_temp,
        avg_salinity, min_depth, max_depth.  All numeric values are
        Python floats or ints (not numpy scalars).
    """
    conditions = []
    params: dict = {}

    if depth_min is not None:
        conditions.append("depth_m >= :depth_min")
        params["depth_min"] = depth_min
    if depth_max is not None:
        conditions.append("depth_m <= :depth_max")
        params["depth_max"] = depth_max
    if lat_min is not None:
        conditions.append("latitude >= :lat_min")
        params["lat_min"] = lat_min
    if lat_max is not None:
        conditions.append("latitude <= :lat_max")
        params["lat_max"] = lat_max
    if lon_min is not None:
        conditions.append("longitude >= :lon_min")
        params["lon_min"] = lon_min
    if lon_max is not None:
        conditions.append("longitude <= :lon_max")
        params["lon_max"] = lon_max

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (
        f"SELECT "
        f"  COUNT(*) AS count, "
        f"  AVG(temp_c) AS avg_temp, "
        f"  MIN(temp_c) AS min_temp, "
        f"  MAX(temp_c) AS max_temp, "
        f"  AVG(salinity) AS avg_salinity, "
        f"  MIN(depth_m) AS min_depth, "
        f"  MAX(depth_m) AS max_depth "
        f"FROM argo_profiles "
        f"{where_clause}".strip()
    )

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text(sql), params).fetchone()

    if row is None:
        return {}

    mapping = dict(row._mapping)
    # Coerce numpy / Decimal types to plain Python primitives
    return {
        k: (float(v) if v is not None and k != "count" else int(v) if v is not None else None)
        for k, v in mapping.items()
    }
