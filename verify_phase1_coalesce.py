import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import db_loader
from api.database import get_engine

def main():
    # Wait, db_loader imports _get_engine from itself. But we updated it to use api.database.get_engine.
    # So we need to make sure we load env.
    from dotenv import load_dotenv
    load_dotenv()
    engine = get_engine()
    
    # 0. Clean up previous test runs if any
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM argo_profiles WHERE float_id = 'TEST_COALESCE';"))

    print("--- 1. Insert row with NULL BGC values ---")
    df1 = pd.DataFrame([{
        "float_id": "TEST_COALESCE",
        "date": datetime(2023, 1, 1, 12, 0, 0),
        "latitude": 0.0,
        "longitude": 0.0,
        "depth_m": 10.0,
        "temp_c": 20.0,
        "salinity": 35.0,
        "oxygen": None,
        "chlorophyll": None,
        "nitrate": None
    }])
    db_loader.save_to_postgres(df1)
    
    with engine.connect() as conn:
        res1 = conn.execute(text("SELECT oxygen, chlorophyll, nitrate FROM argo_profiles WHERE float_id = 'TEST_COALESCE'")).fetchone()
        print(f"State 1: {res1}")
        assert res1 == (None, None, None), "Initial insert should have NULLs"

    print("--- 2. Re-ingest with real BGC values ---")
    df2 = pd.DataFrame([{
        "float_id": "TEST_COALESCE",
        "date": datetime(2023, 1, 1, 12, 0, 0),
        "latitude": 0.0,
        "longitude": 0.0,
        "depth_m": 10.0,
        "temp_c": 20.0,
        "salinity": 35.0,
        "oxygen": 100.0,
        "chlorophyll": 0.5,
        "nitrate": 5.0
    }])
    db_loader.save_to_postgres(df2)
    
    with engine.connect() as conn:
        res2 = conn.execute(text("SELECT oxygen, chlorophyll, nitrate FROM argo_profiles WHERE float_id = 'TEST_COALESCE'")).fetchone()
        print(f"State 2: {res2}")
        assert res2 == (100.0, 0.5, 5.0), "Real values should be updated"

    print("--- 3. Re-ingest with NULL BGC values ---")
    df3 = pd.DataFrame([{
        "float_id": "TEST_COALESCE",
        "date": datetime(2023, 1, 1, 12, 0, 0),
        "latitude": 0.0,
        "longitude": 0.0,
        "depth_m": 10.0,
        "temp_c": 20.0,
        "salinity": 35.0,
        "oxygen": None,
        "chlorophyll": None,
        "nitrate": None
    }])
    db_loader.save_to_postgres(df3)
    
    with engine.connect() as conn:
        res3 = conn.execute(text("SELECT oxygen, chlorophyll, nitrate FROM argo_profiles WHERE float_id = 'TEST_COALESCE'")).fetchone()
        print(f"State 3: {res3}")
        assert res3 == (100.0, 0.5, 5.0), "Previous real values should be preserved when new values are NULL"

    # Clean up
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM argo_profiles WHERE float_id = 'TEST_COALESCE';"))

    print("COALESCE Verification Successful!")

if __name__ == "__main__":
    main()
