import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def generate_report() -> None:
    """
    Loads the processed ARGO parquet dataset and prints a concise
    statistical report to stdout.
    """
    parquet_path = os.getenv("PARQUET_PATH", "data/processed/argo.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    if df.empty:
        print("Data Report: Parquet dataset is empty.")
        return

    # Total rows
    row_count = len(df)

    # Temperature range
    if "temp_c" in df.columns and not df["temp_c"].isna().all():
        temp_min = df["temp_c"].min()
        temp_max = df["temp_c"].max()
        temp_range = f"{temp_min:.2f}°C to {temp_max:.2f}°C"
    else:
        temp_range = "N/A"

    # Salinity range
    if "salinity" in df.columns and not df["salinity"].isna().all():
        sal_min = df["salinity"].min()
        sal_max = df["salinity"].max()
        sal_range = f"{sal_min:.2f} PSU to {sal_max:.2f} PSU"
    else:
        sal_range = "N/A"

    # Depth range
    if "depth_m" in df.columns and not df["depth_m"].isna().all():
        depth_min = df["depth_m"].min()
        depth_max = df["depth_m"].max()
        depth_range = f"{depth_min:.1f}m to {depth_max:.1f}m"
    else:
        depth_range = "N/A"

    # Region breakdown (just list unique values)
    if "ocean_region" in df.columns:
        regions = df["ocean_region"].dropna().unique().tolist()
        regions_str = ", ".join(sorted(regions)) if regions else "None"
    else:
        regions_str = "Feature not extracted"

    # Date range
    if "date" in df.columns:
        valid_dates = df["date"].dropna()
        if not valid_dates.empty:
            date_min = valid_dates.min().strftime("%Y-%m-%d")
            date_max = valid_dates.max().strftime("%Y-%m-%d")
            date_range = f"{date_min} to {date_max}"
        else:
            date_range = "N/A"
    else:
        date_range = "N/A"

    # Print the report
    print("=" * 40)
    print("      SeaBorg Data Report")
    print("=" * 40)
    print(f"Total rows:      {row_count:,}")
    print(f"Date range:      {date_range}")
    print(f"Temp range:      {temp_range}")
    print(f"Salinity range:  {sal_range}")
    print(f"Depth range:     {depth_range}")
    print(f"Ocean Regions:   {regions_str}")
    print("=" * 40)


if __name__ == "__main__":
    generate_report()
