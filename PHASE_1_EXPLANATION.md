# PHASE 1 — Data Ingestion & ETL (SeaBorg)

## What was implemented

Phase 1 ingestion is implemented with local `.nc` files from `data/raw/` (no downloader usage):

- `ingestion/parser.py`
  - Reads one local NetCDF file.
  - Extracts ARGO variables into a tabular DataFrame:
    - `PRES -> depth_m`
    - `TEMP -> temp_c`
    - `PSAL -> salinity`
    - `LATITUDE -> latitude`
    - `LONGITUDE -> longitude`
    - `JULD -> date`
  - Derives `float_id` from file name before the first underscore.
  - Drops rows where `temp_c`, `salinity`, or `depth_m` is missing.
  - Returns `(cleaned_df, original_dataset)`.

- `ingestion/qc_filter.py`
  - Applies ARGO QC logic using `TEMP_QC` and `PSAL_QC`.
  - Keeps only rows where both flags are `1`.
  - Applies scientific range checks:
    - `temp_c`: -3 to 40
    - `salinity`: 20 to 42
    - `depth_m`: > 0
  - Returns the filtered DataFrame.

- `ingestion/db_loader.py`
  - `save_to_postgres(df)` appends rows to `argo_profiles` with `if_exists="append"`.
  - `save_to_parquet(df)` writes to `PARQUET_PATH`, merges with existing data, deduplicates on:
    - `(float_id, date, depth_m)`

- `scripts/run_ingestion.py`
  - Runs the complete ETL for every `.nc` file in `data/raw/`:
    1. parse
    2. QC filter
    3. save to PostgreSQL
    4. save to Parquet
  - Prints per-file row conversion and total ingested rows.

## Why this phase is needed

SeaBorg’s later phases (RAG, LLM, API, charts) depend on structured, validated ocean profile data.
Phase 1 converts raw sensor files into:

- a relational dataset in PostgreSQL (for API and statistics),
- and a Parquet dataset (for fast retrieval/indexing in RAG).

Without this ETL phase, downstream retrieval and analysis cannot be reliable.

## How it works

1. Place `.nc` files in `data/raw/`.
2. Run:

```bash
python scripts/run_ingestion.py
```

3. For each file, the script:
   - parses local NetCDF arrays to rows,
   - removes invalid values using QC flags and range checks,
   - appends clean rows to DB,
   - updates Parquet with deduplication.

4. The script prints:
   - per-file: `raw -> after QC`
   - final total rows ingested.

## Key concepts used

- **xarray** for reading multidimensional NetCDF files.
- **DataFrame normalization** to convert profile/level arrays into row-wise tabular data.
- **Quality control filtering** with ARGO QC flags (`1 = good`).
- **Idempotent Parquet maintenance** using dedup keys to avoid repeated row copies across runs.
- **Append-only DB loading** so historical records are preserved.
