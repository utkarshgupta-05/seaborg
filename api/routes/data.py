import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, Query
from sqlalchemy import create_engine, text

load_dotenv()

router = APIRouter()


def _get_engine():
    """Creates a SQLAlchemy engine from DATABASE_URL."""
    return create_engine(os.getenv("DATABASE_URL"), future=True)


@router.get("/floats")
def list_floats(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """
    Returns a paginated list of unique float IDs with date range and bounding box.

    Args:
        page: Page number (1-based).
        page_size: Number of results per page.

    Returns:
        JSON with total count, page info, and list of float summaries.

    Side effects:
        Queries PostgreSQL.
    """
    engine = _get_engine()
    sql = """
        SELECT
            float_id,
            MIN(date)      AS first_seen,
            MAX(date)      AS last_seen,
            MIN(latitude)  AS lat_min,
            MAX(latitude)  AS lat_max,
            MIN(longitude) AS lon_min,
            MAX(longitude) AS lon_max,
            COUNT(*)       AS record_count
        FROM argo_profiles
        GROUP BY float_id
        ORDER BY float_id
        LIMIT :limit OFFSET :offset
    """
    count_sql = "SELECT COUNT(DISTINCT float_id) AS total FROM argo_profiles"

    with engine.connect() as conn:
        total = conn.execute(text(count_sql)).scalar()
        rows = conn.execute(
            text(sql),
            {"limit": page_size, "offset": (page - 1) * page_size},
        ).fetchall()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "floats": [dict(r._mapping) for r in rows],
    }


@router.get("/float/{float_id}")
def get_float(
    float_id: str,
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    depth_min: Optional[float] = Query(default=None),
    depth_max: Optional[float] = Query(default=None),
):
    """
    Returns all readings for a single float with optional filters.

    Args:
        float_id: The ARGO float identifier.
        start_date: Optional ISO date string filter start.
        end_date: Optional ISO date string filter end.
        depth_min: Optional minimum depth in metres.
        depth_max: Optional maximum depth in metres.

    Returns:
        JSON list of matching rows.

    Side effects:
        Queries PostgreSQL.
    """
    engine = _get_engine()
    conditions = ["float_id = :float_id"]
    params: dict = {"float_id": float_id}

    if start_date:
        conditions.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("date <= :end_date")
        params["end_date"] = end_date
    if depth_min is not None:
        conditions.append("depth_m >= :depth_min")
        params["depth_min"] = depth_min
    if depth_max is not None:
        conditions.append("depth_m <= :depth_max")
        params["depth_max"] = depth_max

    where = " AND ".join(conditions)
    sql = f"SELECT * FROM argo_profiles WHERE {where} ORDER BY date, depth_m"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    return [dict(r._mapping) for r in rows]


@router.get("/stats")
def get_stats():
    """
    Returns aggregate statistics for the full dataset.

    Returns:
        JSON with total_rows, date_range, and geographic_coverage.

    Side effects:
        Queries PostgreSQL.
    """
    engine = _get_engine()
    sql = """
        SELECT
            COUNT(*)            AS total_rows,
            MIN(date)           AS earliest_date,
            MAX(date)           AS latest_date,
            MIN(latitude)       AS lat_min,
            MAX(latitude)       AS lat_max,
            MIN(longitude)      AS lon_min,
            MAX(longitude)      AS lon_max,
            COUNT(DISTINCT float_id) AS unique_floats
        FROM argo_profiles
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).fetchone()

    return dict(row._mapping)