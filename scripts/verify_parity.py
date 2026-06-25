import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def main() -> None:
    load_dotenv()
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set.")
    
    engine = create_engine(database_url, future=True)
    
    parquet_path = os.getenv("PARQUET_PATH", "data/processed/argo.parquet")
    if not Path(parquet_path).exists():
        raise SystemExit(f"Parquet file {parquet_path} does not exist.")
        
    print("Loading Parquet snapshot...")
    pq_df = pd.read_parquet(parquet_path)
    
    print("Loading Postgres metrics...")
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
    
    print("\n--- Parity Check Results ---")
    all_match = True
    for key in pq_metrics:
        pg_val = pg_metrics[key]
        pq_val = pq_metrics[key]
        match_str = "MATCH" if pg_val == pq_val else "MISMATCH"
        if pg_val != pq_val:
            all_match = False
        print(f"{key.ljust(20)}: PG={pg_val:<8} PQ={pq_val:<8} [{match_str}]")
        
    print("\n--- Column Verification ---")
    expected_cols = ["float_id", "date", "latitude", "longitude", "depth_m", "temp_c", "salinity", "oxygen", "chlorophyll", "nitrate"]
    actual_cols = list(pq_df.columns)
    if actual_cols == expected_cols:
        print("Columns exactly match the expected Parquet schema.")
    else:
        print(f"MISMATCH! Expected: {expected_cols}\nActual: {actual_cols}")
        all_match = False
        
    if all_match:
        print("\n[SUCCESS] Verification SUCCESS. Perfect parity achieved.")
        sys.exit(0)
    else:
        print("\n[FAILED] Verification FAILED. Split-brain detected.")
        sys.exit(1)

import sys
if __name__ == "__main__":
    main()
