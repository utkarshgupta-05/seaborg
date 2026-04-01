import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()


def _get_engine():
    """Creates a SQLAlchemy engine from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set in environment/.env.")
    return create_engine(database_url, future=True)


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
        print("Loaded 0 rows into PostgreSQL (empty DataFrame).")
        return

    engine = _get_engine()
    df.to_sql("argo_profiles", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into PostgreSQL.")


def save_to_parquet(df: pd.DataFrame) -> None:
    """
    Saves cleaned rows into a deduplicated Parquet dataset.

    Args:
        df: Cleaned DataFrame with SeaBorg ingestion schema columns.

    Returns:
        None.

    Side effects:
        Creates/updates Parquet file at PARQUET_PATH and prints row count written.
    """
    parquet_path = os.getenv("PARQUET_PATH", "data/processed/argo.parquet")
    target = Path(parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        if not target.exists():
            # Preserve pipeline contract: create an empty parquet only when none exists yet.
            empty_df = pd.DataFrame(columns=["float_id", "date", "latitude", "longitude", "depth_m", "temp_c", "salinity"])
            empty_df.to_parquet(target, index=False)
            print(f"Wrote 0 rows to Parquet: {target}")
        else:
            print(f"Parquet unchanged (empty batch): {target}")
        return

    working_df = df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")

    if target.exists():
        existing_df = pd.read_parquet(target)
        existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
        combined = pd.concat([existing_df, working_df], ignore_index=True)
    else:
        combined = working_df

    deduped = combined.drop_duplicates(subset=["float_id", "date", "depth_m"]).reset_index(drop=True)
    deduped.to_parquet(target, index=False)
    print(f"Wrote {len(deduped)} rows to Parquet: {target}")
