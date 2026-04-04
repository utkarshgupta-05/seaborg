import os

from dotenv import load_dotenv
from fastapi import APIRouter

from api.models import ChatRequest, ChatResponse
from llm.query_engine import answer_query
from rag.retriever import retrieve

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
    Handles POST /chat. Retrieves ARGO context, calls LLM, returns structured response.

    Args:
        req: ChatRequest with message and optional session_id.

    Returns:
        ChatResponse with answer, chart_type, float_ids, sql_used, confidence.

    Side effects:
        Calls FAISS retriever and OpenAI API.
    """
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
    )