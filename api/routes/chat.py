import os
import math
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter

from api.models import ChatRequest, ChatResponse
from llm.query_engine import answer_query
from rag.retriever import retrieve
from router.query_router import classify_query, QueryType
from structured_query.engine import answer_structured_query

from visualisation.map_chart import plot_float_map
from visualisation.profile_chart import plot_depth_profile
from visualisation.timeseries_chart import plot_timeseries

load_dotenv()

router = APIRouter()


def detect_chart_type(message: str) -> str:
    """
    Classifies a user message into a chart type using keyword matching.

    Args:
        message: The user's natural language message.

    Returns:
        One of "map", "profile", "timeseries", or "none".
        First match wins — evaluated in that order.
    """
    msg = message.lower()

    map_keywords = ["where", "map", "location", "region", "ocean", "sea", "coordinates"]
    profile_keywords = ["depth", "profile", "pressure", "meter", "vertical"]
    timeseries_keywords = ["trend", "over time", "monthly", "year", "history", "change"]

    if any(kw in msg for kw in map_keywords):
        return "map"
    if any(kw in msg for kw in profile_keywords):
        return "profile"
    if any(kw in msg for kw in timeseries_keywords):
        return "timeseries"
    return "none"


def detect_visualization_intent(message: str) -> str | None:
    """
    Classifies if the message requests a visualization and returns the type.
    Returns 'map', 'profile', 'timeseries', or None.
    """
    msg = message.lower()
    viz_indicators = ["show", "plot", "draw", "map", "visualize", "graph", "chart", "trend", "over time", "location", "observations"]
    
    has_viz_intent = any(indicator in msg for indicator in viz_indicators)
    if not has_viz_intent:
        return None
        
    map_keywords = ["where", "map", "location", "region", "ocean", "sea", "coordinates", "observations"]
    profile_keywords = ["depth", "profile", "pressure", "meter", "vertical"]
    timeseries_keywords = ["trend", "over time", "monthly", "year", "history", "change"]
    
    if any(kw in msg for kw in map_keywords):
        return "map"
    if any(kw in msg for kw in profile_keywords):
        return "profile"
    if any(kw in msg for kw in timeseries_keywords):
        return "timeseries"
    return None


def detect_variable(message: str) -> str:
    msg = message.lower()
    if "salinity" in msg:
        return "salinity"
    if "oxygen" in msg:
        return "oxygen"
    return "temp_c"


def sanitize_plotly_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_plotly_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_plotly_json(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_plotly_json(x) for x in obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, np.datetime64):
        return str(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return sanitize_plotly_json(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def generate_visualization_payload(message: str, df: pd.DataFrame, float_ids: list[str]):
    viz_type = detect_visualization_intent(message)
    if not viz_type or df.empty:
        return None, None, None, None
        
    variable = detect_variable(message)
    var_title = {
        "temp_c": "Temperature",
        "salinity": "Salinity",
        "oxygen": "Oxygen"
    }.get(variable, variable.capitalize())
    
    float_id = float_ids[0] if float_ids else "Unknown"
    fig = None
    title = None
    description = None
    
    if viz_type == "map":
        msg_lower = message.lower()
        if len(float_ids) == 1:
            title = f"ARGO Float Positions — Float {float_id}"
        elif "atlantic" in msg_lower:
            title = "ARGO Float Positions — Atlantic Ocean"
        elif "indian" in msg_lower:
            title = "ARGO Float Positions — Indian Ocean"
        elif "pacific" in msg_lower:
            title = "ARGO Float Positions — Pacific Ocean"
        else:
            title = "ARGO Float Positions"
        
        fig = plot_float_map(df, title=title)
        description = "Geospatial scatter map of ARGO float observations colored by temperature (°C)."
        
    elif viz_type == "profile":
        title = f"{var_title} Profile — Float {float_id}"
        fig = plot_depth_profile(df, float_id, variable=variable)
        description = f"Vertical profile of {variable} vs depth (m) for float {float_id}."
        
    elif viz_type == "timeseries":
        title = f"{var_title} Over Time — Float {float_id}"
        fig = plot_timeseries(df, float_id, variable=variable)
        description = f"Historical time series of daily/monthly averaged {variable} for float {float_id}."
        
    if fig:
        raw_json = fig.to_plotly_json()
        sanitized_json = sanitize_plotly_json(raw_json)
        return viz_type, sanitized_json, title, description
        
    return None, None, None, None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Handles POST /chat. Routes query dynamically, retrieves data, returns structured response.
    """
    query_type = classify_query(req.message)
    print(f"[ROUTER] {query_type.name}")

    if query_type == QueryType.STRUCTURED:
        result = answer_structured_query(req.message)
        answer = result["summary"]
        chart_type = detect_chart_type(req.message)
        
        rows_df = result["rows"]
        if not rows_df.empty and "float_id" in rows_df.columns:
            float_ids = rows_df["float_id"].unique().tolist()
        else:
            float_ids = []
            
        sql_used = "N/A (Structured Engine)"
        confidence = 1.0
        metadata = result.get("metadata")
        
        viz_type, viz_data, chart_title, chart_description = generate_visualization_payload(
            req.message, rows_df, float_ids
        )
        
        return ChatResponse(
            answer=answer,
            chart_type=chart_type,
            float_ids=float_ids,
            sql_used=sql_used,
            confidence=confidence,
            metadata=metadata,
            visualization_type=viz_type,
            visualization_data=viz_data,
            chart_title=chart_title,
            chart_description=chart_description
        )

    # SEMANTIC PATH (Default)
    rows = retrieve(req.message, top_k=5)
    answer, sql = answer_query(req.message, rows)
    chart_type = detect_chart_type(req.message)
    float_ids = rows["float_id"].unique().tolist()

    viz_type, viz_data, chart_title, chart_description = generate_visualization_payload(
        req.message, rows, float_ids
    )

    return ChatResponse(
        answer=answer,
        chart_type=chart_type,
        float_ids=float_ids,
        sql_used=sql,
        confidence=0.85,
        metadata={"query_type": "semantic"},
        visualization_type=viz_type,
        visualization_data=viz_data,
        chart_title=chart_title,
        chart_description=chart_description
    )