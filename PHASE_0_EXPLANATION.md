# PHASE 0 — Project Scaffold (SeaBorg)

## What was implemented

- The **exact project folder structure** defined in `SEABORG_AGENT_README.md` (Section 3).
- Core configuration files:
  - `requirements.txt`
  - `.gitignore`
  - `.env.example`
  - `.env` (local-only; ignored by git)
- A working database bootstrap script: `scripts/setup_db.py`
  - Creates the `argo_profiles` table
  - Creates the required indexes:
    - `idx_float_id` on `float_id`
    - `idx_date` on `date`
    - `idx_lat_lon` on `(latitude, longitude)`

## Why Phase 0 is needed

SeaBorg is built in multiple phases (ingestion → RAG → LLM → API → frontend). Phase 0 ensures:

- Everyone on the team has the **same directory layout**, so later modules can import each other reliably.
- Environment variables are managed consistently via `.env`, so nothing is hardcoded.
- The PostgreSQL database has the **base table and indexes** ready before any ingestion writes data.

## How it works

### 1) Environment configuration

- `.env.example` shows all keys the project expects, with placeholders.
- `.env` is where you put your real values locally (and it is ignored via `.gitignore`).
- Any module that reads environment variables must call `load_dotenv()` first.

### 2) Database bootstrap (`scripts/setup_db.py`)

When you run:

```bash
python scripts/setup_db.py
```

the script:

- Loads environment variables from `.env`
- Reads `DATABASE_URL`
- Connects to PostgreSQL using SQLAlchemy
- Executes SQL statements that are safe to run multiple times:
  - `CREATE TABLE IF NOT EXISTS ...`
  - `CREATE INDEX IF NOT EXISTS ...`
- Prints **`Database ready.`** if everything succeeds

## Key concepts used (beginner-friendly)

- **Project scaffold**: creating the directories and empty files first so future code has a place to live.
- **Environment variables**: configuration values (like database URLs) are stored outside code so you can change them without editing Python files.
- **SQLAlchemy engine**: a reusable database connection manager for running SQL against PostgreSQL.
- **Idempotent migrations**: using `IF NOT EXISTS` so you can safely re-run setup scripts without breaking your database.

## Phase 0 verification check

Run:

```bash
python scripts/setup_db.py
```

Expected output:

```text
Database ready.
```
