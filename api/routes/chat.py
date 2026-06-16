import os

from dotenv import load_dotenv
from fastapi import APIRouter

from api.models import ChatRequest, ChatResponse
from llm.query_engine import answer_query
from rag.retriever import retrieve

load_dotenv()

router = APIRouter()


from router.query_router import classify_query, QueryType
from structured_query.engine import answer_structured_query

def detect_chart_type(message: str) -> str:
    """
    Classifies a user message into a chart type using keyword matching.

    Args:
        message: The user's natural language message.

    Returns:
        One of "map", "profile", "timeseries", or "none".
        First match wins — evaluated in that order.

    Side effects:
        None.
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
        
        # Defensive guard for float_ids
        rows_df = result["rows"]
        if not rows_df.empty and "float_id" in rows_df.columns:
            float_ids = rows_df["float_id"].unique().tolist()
        else:
            float_ids = []
            
        sql_used = "N/A (Structured Engine)"
        confidence = 1.0  # Structured engine is deterministic
        metadata = result.get("metadata")
        
        return ChatResponse(
            answer=answer,
            chart_type=chart_type,
            float_ids=float_ids,
            sql_used=sql_used,
            confidence=confidence,
            metadata=metadata
        )

    # SEMANTIC PATH (Default)
    rows = retrieve(req.message, top_k=5)
    answer, sql = answer_query(req.message, rows)
    chart_type = detect_chart_type(req.message)
    float_ids = rows["float_id"].unique().tolist()

    return ChatResponse(
        answer=answer,
        chart_type=chart_type,
        float_ids=float_ids,
        sql_used=sql,
        confidence=0.85,
        metadata={"query_type": "semantic"}
    )