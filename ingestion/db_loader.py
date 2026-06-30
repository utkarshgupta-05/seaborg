import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

from api.database import get_engine


from sqlalchemy.dialects.postgresql import insert

def save_to_postgres(df: pd.DataFrame) -> None:
    """
    Appends cleaned rows to PostgreSQL table `argo_profiles`.

    Args:
        df: Cleaned DataFrame with SeaBorg ingestion schema columns.

    Returns:
        None.

    Side effects:
        Writes rows to PostgreSQL and prints row count loaded.
    """
    if df.empty:
        logger.info("Loaded 0 rows into PostgreSQL (empty DataFrame).")
        return

    engine = get_engine()
    
    def insert_on_conflict_nothing(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]
        from sqlalchemy import text
        stmt = insert(table.table).values(data)
        stmt = stmt.on_conflict_do_update(
            index_elements=["float_id", "date", "depth_m"],
            set_={
                "oxygen":      text("COALESCE(EXCLUDED.oxygen, argo_profiles.oxygen)"),
                "chlorophyll":  text("COALESCE(EXCLUDED.chlorophyll, argo_profiles.chlorophyll)"),
                "nitrate":     text("COALESCE(EXCLUDED.nitrate, argo_profiles.nitrate)"),
            }
        )
        conn.execute(stmt)

    df.to_sql("argo_profiles", engine, if_exists="append", index=False, method=insert_on_conflict_nothing)
    logger.info(f"Loaded {len(df)} rows into PostgreSQL.")


def export_parquet_snapshot() -> None:
    """
    Exports a complete, deduplicated Parquet snapshot directly from PostgreSQL.
    
    Returns:
        None.
        
    Side effects:
        Overwrites Parquet file at PARQUET_PATH and prints row count written.
    """
    engine = get_engine()
    
    # Select exactly the columns needed for Parquet schema
    sql = "SELECT float_id, date, latitude, longitude, depth_m, temp_c, salinity, oxygen, chlorophyll, nitrate FROM argo_profiles"
    
    df = pd.read_sql(sql, engine)
    
    if df.empty:
        logger.info("PostgreSQL is empty. Exported 0 rows to Parquet.")
        
    # Ensure correct dtypes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    float_cols = ["latitude", "longitude", "depth_m", "temp_c", "salinity", "oxygen", "chlorophyll", "nitrate"]
    for col in float_cols:
        df[col] = df[col].astype(float)
        
    # Verify exported column list exactly matches expected Parquet schema
    expected_cols = ["float_id", "date", "latitude", "longitude", "depth_m", "temp_c", "salinity", "oxygen", "chlorophyll", "nitrate"]
    if list(df.columns) != expected_cols:
        logger.warning(f"Export column mismatch! Expected {expected_cols}, got {list(df.columns)}")
        
    parquet_path = os.getenv("PARQUET_PATH", "data/processed/argo.parquet")
    target = Path(parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(target, index=False)
    logger.info(f"Exported {len(df)} rows to Parquet snapshot: {target}")
