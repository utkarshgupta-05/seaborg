import json
import time
from pathlib import Path
import pytest

from rag.retriever import load_index
from api.models import ChatRequest
from router.query_router import classify_query
from api.routes.chat import chat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = PROJECT_ROOT / "benchmark" / "queries.json"

def get_queries():
    if not BENCHMARK_FILE.exists():
        return []
    with open(BENCHMARK_FILE, "r") as f:
        return json.load(f)

QUERIES = get_queries()

@pytest.fixture(scope="module", autouse=True)
def setup_faiss():
    try:
        load_index()
    except Exception as e:
        pytest.skip(f"Failed to load FAISS index: {e}")

@pytest.mark.parametrize("query_data", QUERIES, ids=lambda q: f"Query_{q['id']}")
def test_benchmark_route(query_data):
    """
    Tests that the routing logic classifies the benchmark queries exactly
    as expected by the dataset.
    """
    query_text = query_data["query"]
    expected_route = query_data["expected_route"]
    
    actual_route = classify_query(query_text).value
    
    assert actual_route == expected_route, (
        f"Route mismatch for query '{query_text}'. "
        f"Expected {expected_route}, got {actual_route}."
    )

@pytest.mark.parametrize("query_data", QUERIES, ids=lambda q: f"Query_{q['id']}")
def test_benchmark_execution(query_data):
    """
    Executes the query end-to-end to ensure it does not crash
    and returns a response. Note: This can be slow if LLM calls are made.
    """
    query_text = query_data["query"]
    req = ChatRequest(message=query_text)
    
    # We bypass HTTP and directly invoke the chat router
    resp = chat(req)
    
    assert resp is not None
    assert isinstance(resp.response, str)
    assert len(resp.response) > 0
