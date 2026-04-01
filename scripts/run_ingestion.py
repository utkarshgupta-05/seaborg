import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import db_loader, parser, qc_filter


def main() -> None:
    """
    Runs the full local-file ingestion pipeline for all `.nc` files in `data/raw/`.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Reads NetCDF files, writes validated rows to PostgreSQL and Parquet, and prints progress.
    """
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        raise SystemExit("data/raw/ directory does not exist.")

    nc_files = sorted(raw_dir.glob("*.nc"))
    if not nc_files:
        print("No .nc files found in data/raw/. Nothing to ingest.")
        return

    total_rows = 0

    for filepath in nc_files:
        df, dataset = parser.parse_netcdf(str(filepath))
        clean_df = qc_filter.apply_qc(df, dataset)
        dataset.close()

        db_loader.save_to_postgres(clean_df)
        db_loader.save_to_parquet(clean_df)

        print(f"{filepath}: {len(df)} raw -> {len(clean_df)} after QC")
        total_rows += len(clean_df)

    print(f"Total rows ingested: {total_rows}")


if __name__ == "__main__":
    main()
