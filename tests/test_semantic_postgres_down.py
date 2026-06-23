import asyncio
import os
import sys
from unittest.mock import patch
from api.routes.chat import chat
from pydantic import BaseModel

class MockReq(BaseModel):
    message: str

def mock_get_engine():
    raise Exception("Simulated OperationalError: Postgres is down!")

from rag.retriever import load_index
load_index()

@patch('structured_query.repository.get_engine', side_effect=mock_get_engine)
@patch('structured_query.repository._availability_cache', {}) # Clear cache
def test_postgres_down(mock_engine):
    print("Testing Postgres down with semantic query...")
    req = MockReq(message="what is ocean temperature?")
    res = chat(req)
    print("Semantic Query Result:", res.answer[:50], "...")
    print("Confidence:", res.confidence)
    
    print("\nTesting Postgres down with hybrid query...")
    req2 = MockReq(message="explain salinity trends")
    res2 = chat(req2)
    print("Hybrid Query Result:", res2.answer[:50], "...")
    print("Confidence:", res2.confidence)

if __name__ == "__main__":
    test_postgres_down()
