import pytest
import pandas as pd
import numpy as np
import faiss
import os

from rag import retriever
from llm.geo_mapping import REGION_BOUNDS
from router.query_router import _GEO_REGIONS

def test_schema_columns():
    """Assert retrieve() returns chlorophyll and nitrate columns."""
    df = pd.DataFrame({"id": [1]})
    ensured = retriever._ensure_schema(df)
    assert "chlorophyll" in ensured.columns
    assert "nitrate" in ensured.columns
    assert "oxygen" in ensured.columns

def test_non_null_round_trip():
    """Ensure missing columns are filled with pd.NA."""
    df = pd.DataFrame({"temp_c": [20.0]})
    ensured = retriever._ensure_schema(df)
    assert pd.isna(ensured["chlorophyll"].iloc[0])

def test_preference_rerank():
    """Preference rerank test with synthetic FAISS index."""
    # Mock data with one sparse row having nitrate and one missing nitrate
    df = pd.DataFrame({
        "float_id": ["A", "B"],
        "date": ["2023-01-01", "2023-01-02"],
        "latitude": [0.0, 0.0],
        "longitude": [0.0, 0.0],
        "depth_m": [10.0, 10.0],
        "temp_c": [20.0, 20.0],
        "salinity": [35.0, 35.0],
        "oxygen": [100.0, None],
        "chlorophyll": [0.5, None],
        "nitrate": [None, 5.0] # B has nitrate, A has None
    })
    
    # Create fake index
    index = faiss.IndexFlatL2(384)
    # Embeddings for A and B
    vecs = np.random.rand(2, 384).astype("float32")
    index.add(vecs)
    
    # Monkeypatch
    retriever._df = df
    retriever._index = index
    
    # Query for nitrate (sparse variable)
    res = retriever.retrieve("where is nitrate", top_k=2, variable="nitrate")
    
    assert res.iloc[0]["float_id"] == "B", "Row with nitrate should be preferred"
    assert res.iloc[1]["float_id"] == "A", "Row with NULL nitrate should be ranked lower"

def test_region_sync():
    """Every entry in _GEO_REGIONS has matching bounds or is just a short alias of one."""
    for region in _GEO_REGIONS:
        assert region in REGION_BOUNDS or region in ["atlantic", "indian", "pacific", "mediterranean"]
