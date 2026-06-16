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