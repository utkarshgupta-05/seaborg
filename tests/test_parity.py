import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pytest

load_dotenv()

@pytest.fixture(scope="module")
def parity_data():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set.")
        
    engine = create_engine(database_url, future=True)
    
    parquet_path = os.getenv("PARQUET_PATH", "data/processed/argo.parquet")
    if not Path(parquet_path).exists():
        pytest.skip(f"Parquet file {parquet_path} does not exist.")
        
    pq_df = pd.read_parquet(parquet_path)
    
    sql = """
    SELECT 
        COUNT(*) as total_rows,
        COUNT(salinity) as salinity_count,
        COUNT(oxygen) as oxygen_count,
        COUNT(chlorophyll) as chlorophyll_count,
        COUNT(nitrate) as nitrate_count
    FROM argo_profiles;
    """
    with engine.connect() as conn:
        pg_metrics = dict(conn.execute(text(sql)).fetchone()._mapping)
        
    pq_metrics = {
        "total_rows": len(pq_df),
        "salinity_count": pq_df["salinity"].notna().sum(),
        "oxygen_count": pq_df["oxygen"].notna().sum(),
        "chlorophyll_count": pq_df["chlorophyll"].notna().sum(),
        "nitrate_count": pq_df["nitrate"].notna().sum()
    }
    
    return pg_metrics, pq_metrics, pq_df

def test_total_rows_parity(parity_data):
    pg, pq, _ = parity_data
    assert pg["total_rows"] == pq["total_rows"], "Total row counts do not match"

def test_salinity_count_parity(parity_data):
    pg, pq, _ = parity_data
    assert pg["salinity_count"] == pq["salinity_count"], "Salinity non-null counts do not match"

def test_oxygen_count_parity(parity_data):
    pg, pq, _ = parity_data
    assert pg["oxygen_count"] == pq["oxygen_count"], "Oxygen non-null counts do not match"

def test_chlorophyll_count_parity(parity_data):
    pg, pq, _ = parity_data
    assert pg["chlorophyll_count"] == pq["chlorophyll_count"], "Chlorophyll non-null counts do not match"

def test_nitrate_count_parity(parity_data):
    pg, pq, _ = parity_data
    assert pg["nitrate_count"] == pq["nitrate_count"], "Nitrate non-null counts do not match"

def test_parquet_schema(parity_data):
    _, _, pq_df = parity_data
    expected_cols = ["float_id", "date", "latitude", "longitude", "depth_m", "temp_c", "salinity", "oxygen", "chlorophyll", "nitrate"]
    actual_cols = list(pq_df.columns)
    assert actual_cols == expected_cols, f"Parquet columns mismatch! Expected: {expected_cols}, Actual: {actual_cols}"
