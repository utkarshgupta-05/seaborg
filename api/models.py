from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    chart_type: Literal["map", "profile", "timeseries", "none"]
    float_ids: list[str]
    sql_used: str | None
    confidence: float
    metadata: dict | None = None
    visualization_type: str | None = None
    visualization_data: dict | None = None
    chart_title: str | None = None
    chart_description: str | None = None


class FloatDataRequest(BaseModel):
    float_id: str
    start_date: str | None = None
    end_date: str | None = None
    depth_min: float | None = None
    depth_max: float | None = None


class ExportRequest(BaseModel):
    float_ids: list[str]
    format: Literal["csv", "netcdf"]
    start_date: str | None = None
    end_date: str | None = None


class FloatSummary(BaseModel):
    float_id: str
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    record_count: int


class FloatListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    floats: list[FloatSummary]


class DatasetStats(BaseModel):
    total_rows: int
    earliest_date: datetime | None = None
    latest_date: datetime | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    unique_floats: int