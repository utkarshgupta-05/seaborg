import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)


load_dotenv()


def _get_engine():
    """Creates a SQLAlchemy engine from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set in environment/.env.")
    return create_engine(database_url, future=True)


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

    engine = _get_engine()
    
    def insert_on_conflict_nothing(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data).on_conflict_do_nothing(
            index_elements=["float_id", "date", "depth_m"]
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
    engine = _get_engine()
    
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
