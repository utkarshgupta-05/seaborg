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
    has_errors = False

    for filepath in nc_files:
        try:
            df, dataset = parser.parse_netcdf(str(filepath))
            try:
                clean_df = qc_filter.apply_qc(df, dataset)
                
                if clean_df.empty:
                    print(f"{filepath}: No valid rows after QC.")
                    continue

                db_loader.save_to_postgres(clean_df)

                print(f"{filepath}: {len(df)} raw -> {len(clean_df)} after QC")
                total_rows += len(clean_df)
            finally:
                dataset.close()
        except ValueError as e:
            print(f"Skipping {filepath}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            has_errors = True
            continue

    print(f"Total rows ingested: {total_rows}")
    
    if not has_errors:
        print("Exporting Parquet snapshot...")
        db_loader.export_parquet_snapshot()
    else:
        print("Skipping Parquet export due to processing errors.")


if __name__ == "__main__":
    main()
