"""
tests/test_rag_quality.py

Test suite for Phase 10 RAG Quality & Hallucination Control.
Verifies distance thresholding, confidence calculation, safe fallbacks,
and weak-context rejection.
"""


import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from rag.retriever import retrieve
from retrieval.hybrid_service import hybrid_answer
from api.routes.chat import chat, ChatRequest


# ── Retriever Threshold Tests ─────────────────────────────────────────────────

@patch("rag.retriever._index")
@patch("rag.retriever._df")
@patch("rag.retriever.embed_query")
def test_retriever_applies_distance_threshold(mock_embed, mock_df, mock_index):
    # Mock FAISS returning 3 rows with varying distances
    mock_index.search.return_value = (
        [[0.5, 1.2, 2.0]],  # distances
        [[0, 1, 2]]         # indices
    )
    
    mock_df.iloc.__getitem__.return_value = pd.DataFrame({
        "id": [1, 2, 3],
        "float_id": ["A", "B", "C"],
        "temp_c": [1.0, 2.0, 3.0]
    })
    mock_df.__len__.return_value = 3

    # With threshold 1.5, the row with distance 2.0 should be dropped
    rows = retrieve("test query", top_k=3, distance_threshold=1.5)
    
    assert len(rows) == 2
    assert "faiss_distance" in rows.columns
    assert list(rows["faiss_distance"]) == [0.5, 1.2]


@patch("rag.retriever._index")
@patch("rag.retriever._df")
@patch("rag.retriever.embed_query")
def test_retriever_returns_empty_when_all_weak(mock_embed, mock_df, mock_index):
    # Mock FAISS returning only weak matches
    mock_index.search.return_value = (
        [[2.5, 3.0, 3.5]],  # distances all > 1.5
        [[0, 1, 2]]
    )
    
    mock_df.iloc.__getitem__.return_value = pd.DataFrame({
        "id": [1, 2, 3]
    })
    mock_df.__len__.return_value = 3

    rows = retrieve("nonsense query", top_k=3, distance_threshold=1.5)
    
    assert len(rows) == 0


# ── Hybrid Service Fallback Tests ─────────────────────────────────────────────

@patch("retrieval.hybrid_service.retrieve")
@patch("retrieval.hybrid_service.answer_structured_query")
@patch("retrieval.hybrid_service.Groq")
def test_hybrid_service_low_confidence_when_semantic_empty(mock_groq, mock_structured, mock_retrieve):
    mock_structured.return_value = {
        "summary": "Structured Summary Data",
        "rows": pd.DataFrame({"float_id": ["S1"], "date": ["2020"], "depth_m": [10]}),
        "metadata": {}
    }
    
    # Semantic retriever returns nothing (weak context rejected)
    mock_retrieve.return_value = pd.DataFrame()
    
    mock_llm_instance = MagicMock()
    mock_groq.return_value = mock_llm_instance
    mock_llm_instance.chat.completions.create.return_value.choices[0].message.content = "Answer"

    result = hybrid_answer("some query")
    
    # Confidence should drop to 0.70 because semantic context was empty
    assert result["confidence"] == 0.70
    
    # The LLM prompt should contain the "No supporting records" fallback
    call_args = mock_llm_instance.chat.completions.create.call_args[1]
    prompt_sent = call_args["messages"][0]["content"]
    assert "No supporting records retrieved" in prompt_sent


# ── Chat Route Semantic Fallback Tests ────────────────────────────────────────

@patch("api.routes.chat.retrieve")
@patch("api.routes.chat.route_query")
def test_chat_semantic_safe_fallback(mock_route, mock_retrieve):
    from router.query_router import RoutingResult, QueryType
    mock_route.return_value = RoutingResult(
        intent=QueryType.SEMANTIC,
        structured_signals=[],
        semantic_signals=["semantic(explain)"]
    )
    
    # Mock retrieve returning empty dataframe (weak matches filtered out)
    mock_retrieve.return_value = pd.DataFrame()

    req = ChatRequest(message="nonsense query explain")
    response = chat(req)

    # Should safely fallback without calling LLM
    assert response.confidence == 0.20
    assert "I couldn't find any relevant oceanographic records" in response.answer
    assert response.chart_type == "none"
