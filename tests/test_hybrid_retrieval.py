"""
tests/test_hybrid_retrieval.py

Test suite for the Phase 9 context-fusion hybrid retrieval architecture.
Covers prompt construction, row deduplication, service orchestration,
and compatibility with the ChatResponse schema.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from llm.context_builder import build_hybrid_prompt
from retrieval.hybrid_service import hybrid_answer
from api.models import ChatResponse
from api.routes.chat import chat, ChatRequest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_structured_rows():
    return pd.DataFrame({
        "float_id": ["F1", "F2"],
        "date": ["2023-01-01", "2023-01-02"],
        "depth_m": [500.0, 500.0],
        "temp_c": [4.0, 4.2],
        "salinity": [34.5, 34.6],
        "oxygen": [200.0, 205.0]
    })


@pytest.fixture
def mock_semantic_rows():
    return pd.DataFrame({
        "float_id": ["F2", "F3"],  # F2 overlaps with structured
        "date": ["2023-01-02", "2023-01-03"],
        "depth_m": [500.0, 1000.0],
        "temp_c": [4.2, 2.5],
        "salinity": [34.6, 34.8],
        "oxygen": [205.0, 150.0]
    })


@pytest.fixture
def mock_structured_summary():
    return "Found 2 matching observations.\nAverage temperature: 4.10°C"


# ── Context Builder Tests ─────────────────────────────────────────────────────

def test_build_hybrid_prompt_includes_both_contexts(mock_structured_summary, mock_semantic_rows):
    question = "average temperature at 500m and explain why"
    prompt = build_hybrid_prompt(question, mock_structured_summary, mock_semantic_rows)

    # Must contain the structured summary exactly
    assert "AUTHORITATIVE DATA SUMMARY:" in prompt
    assert mock_structured_summary in prompt

    # Must contain semantic row details
    assert "SUPPORTING RECORDS:" in prompt
    assert "Float F2" in prompt
    assert "Float F3" in prompt

    # Must contain the question
    assert f"Question: {question}" in prompt


def test_build_hybrid_prompt_handles_empty_semantic(mock_structured_summary):
    question = "average temperature at 500m and explain why"
    prompt = build_hybrid_prompt(question, mock_structured_summary, pd.DataFrame())

    assert mock_structured_summary in prompt
    assert "No supporting records retrieved" in prompt


# ── Hybrid Service Tests (Orchestration & Deduplication) ──────────────────────

@patch("retrieval.hybrid_service.Groq")
@patch("retrieval.hybrid_service.retrieve")
@patch("retrieval.hybrid_service.answer_structured_query")
def test_hybrid_service_orchestration(
    mock_structured, mock_retrieve, mock_groq,
    mock_structured_rows, mock_semantic_rows, mock_structured_summary
):
    # Mock structured engine
    mock_structured.return_value = {
        "summary": mock_structured_summary,
        "rows": mock_structured_rows,
        "metadata": {"record_count": 2}
    }

    # Mock semantic retriever
    mock_retrieve.return_value = mock_semantic_rows

    # Mock LLM
    mock_llm_instance = MagicMock()
    mock_groq.return_value = mock_llm_instance
    mock_llm_instance.chat.completions.create.return_value.choices[0].message.content = "Mock LLM Answer"

    result = hybrid_answer("average temperature at 500m and explain why")

    # Verify return shape
    assert "summary" in result
    assert "rows" in result
    assert "metadata" in result
    assert "sql" in result

    assert result["summary"] == "Mock LLM Answer"
    assert result["metadata"]["query_type"] == "hybrid"

    # Verify deduplication: F1, F2 (structured), F3 (semantic). 
    # F2 overlaps so there should be exactly 3 rows total.
    combined_df = result["rows"]
    assert len(combined_df) == 3
    assert set(combined_df["float_id"].tolist()) == {"F1", "F2", "F3"}

    # Verify LLM was called with fused prompt
    mock_llm_instance.chat.completions.create.assert_called_once()
    call_args = mock_llm_instance.chat.completions.create.call_args[1]
    prompt_sent = call_args["messages"][0]["content"]
    assert mock_structured_summary in prompt_sent


# ── API Endpoint Compatibility Tests ──────────────────────────────────────────



@patch("api.routes.chat.hybrid_answer")
@patch("api.routes.chat.route_query")
def test_chat_endpoint_hybrid_route(mock_route, mock_hybrid):
    # Mock router to return HYBRID
    from router.query_router import RoutingResult, QueryType
    mock_route.return_value = RoutingResult(
        intent=QueryType.HYBRID,
        structured_signals=["stats(average)"],
        semantic_signals=["semantic(explain)"]
    )

    # Mock hybrid service output
    mock_hybrid.return_value = {
        "summary": "This is a fused answer.",
        "rows": pd.DataFrame({"float_id": ["F99"], "date": ["2023-01-01"], "depth_m": [500.0]}),
        "metadata": {"query_type": "hybrid"},
        "sql": "SELECT * FROM argo_profiles"
    }

    req = ChatRequest(message="average temperature and explain")
    response = chat(req)

    # Must be valid ChatResponse Pydantic model
    assert isinstance(response, ChatResponse)
    
    assert response.answer == "This is a fused answer."
    assert response.float_ids == ["F99"]
    assert response.metadata["query_type"] == "hybrid"
    assert "routing_signals" in response.metadata
    
    mock_hybrid.assert_called_once_with("average temperature and explain")
