import io
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text

from api.models import ExportRequest

load_dotenv()

router = APIRouter()


def _get_engine():
    """Creates a SQLAlchemy engine from DATABASE_URL."""
    return create_engine(os.getenv("DATABASE_URL"), future=True)


def _query_data(req: ExportRequest) -> pd.DataFrame:
    """
    Queries argo_profiles filtered by float_ids and optional date range.

    Args:
        req: ExportRequest with float_ids, format, and optional date filters.

    Returns:
        DataFrame of matching rows.

    Side effects:
        Queries PostgreSQL.
    """
    engine = _get_engine()
    conditions = ["float_id = ANY(:float_ids)"]
    params: dict = {"float_ids": req.float_ids}

    if req.start_date:
        conditions.append("date >= :start_date")
        params["start_date"] = req.start_date
    if req.end_date:
        conditions.append("date <= :end_date")
        params["end_date"] = req.end_date

    where = " AND ".join(conditions)
    sql = f"SELECT * FROM argo_profiles WHERE {where} ORDER BY float_id, date, depth_m"

    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


@router.post("/export")
def export_data(req: ExportRequest):
    """
    Streams a file download of ARGO data in CSV or NetCDF format.

    Args:
        req: ExportRequest specifying float_ids, format, and optional date range.

    Returns:
        StreamingResponse with appropriate Content-Type and Content-Disposition headers.

    Side effects:
        Queries PostgreSQL and streams file bytes to the client.
    """
    df = _query_data(req)

    if req.format == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="seaborg_export.csv"'
            },
        )

    if req.format == "netcdf":
        import xarray as xr
        ds = xr.Dataset.from_dataframe(df.set_index(["float_id", "date"]))
        buffer = io.BytesIO()
        ds.to_netcdf(buffer)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.read()]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="seaborg_export.nc"'
            },
        )