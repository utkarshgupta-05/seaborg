# SeaBorg — AI-Powered Ocean Data Chatbot
## Complete Agent Build Specification

> **AGENT INSTRUCTIONS:** Read this entire document before writing any code.
> Execute each phase in order. Do not proceed to the next phase until the
> current phase's verification check passes. Every code decision not
> explicitly stated here is yours to make — but all explicit instructions
> are non-negotiable.

**Team:** Shubham Kumar (24BCS237) · Shubham Kulkarni (24BCS236) · Pushkar Trivedi (24BCS203) · Nishkarsh Sharma (24BCS176) · Utkarsh Gupta (24BCS256)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [End-to-End Architecture](#2-end-to-end-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Tech Stack (Fixed — No Substitutions)](#4-tech-stack-fixed--no-substitutions)
5. [Environment Variables](#5-environment-variables)
6. [Database Schema](#6-database-schema)
7. [Phase 0 — Project Scaffold](#phase-0--project-scaffold)
8. [Phase 1 — Data Ingestion & ETL](#phase-1--data-ingestion--etl)
9. [Phase 2 — RAG Pipeline & Vector Index](#phase-2--rag-pipeline--vector-index)
10. [Phase 3 — LLM Integration](#phase-3--llm-integration)
11. [Phase 4 — FastAPI Backend](#phase-4--fastapi-backend)
12. [Phase 5 — Visualisation Engine](#phase-5--visualisation-engine)
13. [Phase 6 — Streamlit Frontend](#phase-6--streamlit-frontend)
14. [Phase 7 — Tests](#phase-7--tests)
15. [Phase 8 — Deployment](#phase-8--deployment)
16. [Data Flow Contracts](#16-data-flow-contracts)
17. [Global Rules](#17-global-rules)

---

## 1. System Overview

SeaBorg lets any user query ARGO oceanographic float data using plain English and receive:
- A written AI answer grounded in real retrieved records
- An interactive Plotly chart (map / depth profile / time series)
- A downloadable CSV or NetCDF export of the underlying data

**Data source:** ARGO float programme — robotic ocean sensors that record temperature,
salinity, and pressure while ascending from depth. Data is distributed as NetCDF files.

**Target users:** Oceanographers, climate scientists, university students, policy makers
who need ocean data insights without writing code or processing NetCDF files.

---

## 2. End-to-End Architecture

### Offline Pipeline (run once before launch)

```
data/raw/*.nc
    → ingestion/parser.py          (parse NetCDF → DataFrame + xarray Dataset)
    → ingestion/qc_filter.py       (apply QC flags + range checks → clean DataFrame)
    → ingestion/db_loader.py       (write → PostgreSQL table + Parquet file)
    → rag/summariser.py            (DataFrame rows → English sentences)
    → rag/embedder.py              (sentences → float32 numpy vectors, shape (n,384))
    → rag/indexer.py               (vectors → FAISS IndexFlatL2 saved to disk)
```

### Runtime Pipeline (every user message)

```
User message (Streamlit)
    → POST /api/chat (FastAPI)
    → rag/retriever.py             (embed query → FAISS search → top-5 Parquet rows)
    → llm/query_engine.py          (rows + question → GPT-4 prompt → answer + SQL)
    → api/routes/chat.py           (detect chart_type → build ChatResponse JSON)
    → Streamlit frontend           (render answer + Plotly chart side by side)
```

**Key architectural constraint:** The LLM never queries the database directly.
It only reads the small retrieved context window. This is the RAG pattern that prevents hallucination.

---

## 3. Folder Structure

Create this exact structure first. Every folder and file listed here must exist.

```
seaborg/
├── README.md
├── requirements.txt
├── .env                          ← never commit; listed in .gitignore
├── .env.example                  ← commit this; placeholder values only
├── .gitignore
│
├── data/
│   ├── raw/                      ← store downloaded .nc files here
│   ├── processed/                ← Parquet files written here
│   └── exports/                  ← CSV / NetCDF user exports written here
│
├── ingestion/
│   ├── __init__.py
│   ├── downloader.py
│   ├── parser.py
│   ├── qc_filter.py
│   └── db_loader.py
│
├── rag/
│   ├── __init__.py
│   ├── summariser.py
│   ├── embedder.py
│   ├── indexer.py
│   └── retriever.py
│
├── llm/
│   ├── __init__.py
│   ├── prompts.py
│   ├── query_engine.py
│   └── nl_to_sql.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── tools.py
│   └── routes/
│       ├── __init__.py
│       ├── chat.py
│       ├── data.py
│       └── export.py
│
├── visualisation/
│   ├── __init__.py
│   ├── map_chart.py
│   ├── profile_chart.py
│   ├── timeseries_chart.py
│   └── exporter.py
│
├── frontend/
│   ├── app.py
│   └── components/
│       ├── __init__.py
│       ├── chat_panel.py
│       ├── chart_panel.py
│       └── sidebar.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_charts.py
│
├── notebooks/
│   ├── 01_explore_netcdf.ipynb
│   ├── 02_test_rag.ipynb
│   └── 03_visualisation_demo.ipynb
│
├── scripts/
│   ├── setup_db.py
│   ├── run_ingestion.py
│   └── build_index.py
│
├── indexes/
│   └── .gitkeep
│
└── models/
    └── .gitkeep
```

---

## 4. Tech Stack (Fixed — No Substitutions)

| Category | Library / Tool | Notes |
|---|---|---|
| NetCDF parsing | `xarray`, `netCDF4` | Open `.nc` files |
| Tabular data | `pandas`, `numpy` | All DataFrame ops |
| Parquet I/O | `pyarrow` | Read/write Parquet |
| Database | `PostgreSQL` + `SQLAlchemy` + `psycopg2-binary` | ORM layer |
| Embeddings | `sentence-transformers` model `all-MiniLM-L6-v2` | 384-dim, local, no GPU |
| Vector search | `faiss-cpu` | Use `chromadb` only if FAISS won't install |
| LLM client | `openai` Python library targeting GPT-4 | Model name from env var |
| Backend | `FastAPI` + `uvicorn` | ASGI server |
| Schemas | `pydantic` v2 | All request/response models |
| Charts | `plotly` | All three chart types; no matplotlib |
| Frontend | `streamlit` | Pure Python; no React/HTML/JS files |
| Testing | `pytest` + `httpx` | `httpx` for async API tests |

---

## 5. Environment Variables

### `.env` (never committed)

```env
DATABASE_URL=postgresql://user:password@localhost:5432/seaborg
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...          # optional but recommended
ENVIRONMENT=development           # or "production"
FAISS_INDEX_PATH=indexes/argo.faiss
PARQUET_PATH=data/processed/argo.parquet
LLM_MODEL=gpt-4                   # change here only; never hardcode elsewhere
API_URL=http://localhost:8000     # frontend uses this to call the backend
```

### `.env.example` (committed to git)

Same keys with placeholder values and a one-line comment per key explaining its purpose.

### `.gitignore` must exclude

```
.env
venv/
__pycache__/
*.pyc
indexes/*.faiss
models/
data/raw/
data/processed/
data/exports/
```

### Rule for all modules

Every module that reads env vars must call `load_dotenv()` at the top before `os.getenv()`. No hardcoded values anywhere.

---

## 6. Database Schema

**Database name:** `seaborg`
**Table name:** `argo_profiles`

| Column | Type | Constraints |
|---|---|---|
| `id` | `SERIAL` | PRIMARY KEY |
| `float_id` | `VARCHAR(20)` | NOT NULL |
| `date` | `TIMESTAMP` | NOT NULL |
| `latitude` | `FLOAT` | range −90 to 90 |
| `longitude` | `FLOAT` | range −180 to 180 |
| `depth_m` | `FLOAT` | > 0 |
| `temp_c` | `FLOAT` | — |
| `salinity` | `FLOAT` | — |
| `oxygen` | `FLOAT` | NULLABLE |
| `created_at` | `TIMESTAMP` | DEFAULT NOW() |

**Indexes to create:**
- `idx_float_id` on `float_id`
- `idx_date` on `date`
- `idx_lat_lon` composite on `(latitude, longitude)`

All CREATE TABLE / CREATE INDEX statements must use `IF NOT EXISTS` so `setup_db.py` is safe to re-run.

---

## Phase 0 — Project Scaffold

**Goal:** Empty but complete directory tree; config files; working DB connection.

### Steps

1. Create every folder and file listed in Section 3. All `__init__.py` files are empty.
2. Create `.env`, `.env.example`, `.gitignore`, `requirements.txt` per Sections 4 & 5.
3. Write `scripts/setup_db.py`:
   - Read `DATABASE_URL` from env.
   - Connect with SQLAlchemy.
   - Create `argo_profiles` table and all three indexes using `IF NOT EXISTS`.
   - Print `"Database ready."` on success.
4. Run `python scripts/setup_db.py` and confirm no errors.

### ✅ Verification

```
python scripts/setup_db.py
# Expected output: "Database ready."
```

---

## Phase 1 — Data Ingestion & ETL

**Goal:** Transform raw `.nc` files into clean rows in PostgreSQL and a Parquet file.

---

### `ingestion/downloader.py`

**Single job:** Download ARGO NetCDF files from EuroArgo FTP server to `data/raw/`.

**Public interface:**
```python
def download_floats(float_ids: list[str]) -> None:
    """
    Download profile NetCDF files for each float_id in float_ids.
    FTP base: ftp://ftp.ifremer.fr/ifremer/argo
    Skip files already present in data/raw/.
    Print download progress per file.
    """
```

---

### `ingestion/parser.py`

**Single job:** Open one `.nc` file → extract required variables → return cleaned DataFrame + xarray Dataset.

**Variables to extract:**

| NetCDF variable | Rename to | Notes |
|---|---|---|
| `PRES` | `depth_m` | Pressure used as depth proxy |
| `TEMP` | `temp_c` | Temperature °C |
| `PSAL` | `salinity` | Practical Salinity Units |
| `LATITUDE` | `latitude` | Decimal degrees |
| `LONGITUDE` | `longitude` | Decimal degrees |
| `JULD` | `date` | Julian date → convert to datetime |

**Additional processing:**
- Derive `float_id` from filename: take the portion before the first underscore.
- Drop rows where `temp_c`, `salinity`, or `depth_m` is NaN.

**Public interface:**
```python
def parse_netcdf(filepath: str) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Returns (cleaned_df, original_dataset).
    The dataset is passed to qc_filter; the df columns must match the DB schema.
    """
```

---

### `ingestion/qc_filter.py`

**Single job:** Apply ARGO QC flags and range checks; return only scientifically valid rows.

**QC flag rule:** ARGO files contain `TEMP_QC` and `PSAL_QC` arrays. Value `1` = good. Keep only rows where both equal `1`.

**Range checks — remove rows outside these bounds:**

| Field | Min | Max |
|---|---|---|
| `temp_c` | −3 °C | 40 °C |
| `salinity` | 20 PSU | 42 PSU |
| `depth_m` | 0 m (exclusive) | ∞ |

**Public interface:**
```python
def apply_qc(df: pd.DataFrame, dataset: xr.Dataset) -> pd.DataFrame:
    """
    Accepts the DataFrame and xarray Dataset from parser.parse_netcdf().
    Returns filtered DataFrame with only scientifically valid rows.
    """
```

---

### `ingestion/db_loader.py`

**Single job:** Persist clean DataFrame to PostgreSQL and Parquet.

**PostgreSQL function:**
- Append rows using `if_exists="append"`.
- Never truncate existing data.
- Print row count loaded.

**Parquet function:**
- Save to `PARQUET_PATH` from env.
- If file already exists: read existing → concat new data → deduplicate on `(float_id, date, depth_m)` → overwrite.
- This keeps Parquet as the single source of truth for the RAG pipeline.

**Public interface:**
```python
def save_to_postgres(df: pd.DataFrame) -> None: ...
def save_to_parquet(df: pd.DataFrame) -> None: ...
```

---

### `scripts/run_ingestion.py`

**Single job:** Orchestrate the full ETL pipeline in one command.

**Logic:**
```
for each .nc file in data/raw/:
    df, dataset = parser.parse_netcdf(filepath)
    clean_df = qc_filter.apply_qc(df, dataset)
    db_loader.save_to_postgres(clean_df)
    db_loader.save_to_parquet(clean_df)
    print(f"{filepath}: {len(df)} raw → {len(clean_df)} after QC")

print(f"Total rows ingested: {total}")
```

### ✅ Verification

```
# Place at least 3 .nc files in data/raw/ first
python scripts/run_ingestion.py
# Expected: per-file row counts + total + no errors
# Confirm: psql seaborg -c "SELECT COUNT(*) FROM argo_profiles;"
# Confirm: data/processed/argo.parquet exists
```

---

## Phase 2 — RAG Pipeline & Vector Index

**Goal:** Make every data row semantically searchable via FAISS.

---

### `rag/summariser.py`

**Single job:** Convert one data row into a natural English sentence.

**Required sentence format:**
> "Float [float_id] recorded a temperature of [temp_c]°C and salinity of [salinity] PSU at [depth_m]m depth on [YYYY-MM-DD] at coordinates ([lat], [lon])."

Round: lat/lon to 2 decimal places, depth to nearest metre, temp to 1 decimal place, salinity to 2 decimal places.

**Public interface:**
```python
def summarise_row(row: dict | pd.Series) -> str:
    """Returns a single English sentence capturing all key oceanographic values."""
```

---

### `rag/embedder.py`

**Single job:** Load `all-MiniLM-L6-v2` once (singleton); provide batch + single embed functions.

**Rules:**
- Load model lazily on first call — importing this module must not trigger a download.
- Both functions return `numpy float32` arrays.

**Public interface:**
```python
def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns shape (n, 384) float32 array."""

def embed_query(query: str) -> np.ndarray:
    """Returns shape (1, 384) float32 array."""
```

---

### `rag/indexer.py`

**Single job:** Build and save the FAISS index from the full Parquet file.

**Logic:**
```
1. Read Parquet file at PARQUET_PATH
2. For each row → summariser.summarise_row() → collect all summaries
3. embedder.embed_texts(all_summaries) → vectors shape (n, 384)
4. faiss.IndexFlatL2(384) → index.add(vectors)
5. faiss.write_index(index, FAISS_INDEX_PATH)
6. Print: "Indexed {n} vectors → saved to {FAISS_INDEX_PATH}"
```

This module has no public function — it is invoked as a script by `scripts/build_index.py`.

---

### `rag/retriever.py`

**Single job:** Load the FAISS index + Parquet once; expose a `retrieve()` function.

**Module-level state:**
```python
_index = None      # FAISS index, loaded once
_df    = None      # Full Parquet DataFrame, loaded once
```

**Public interface:**
```python
def load_index() -> None:
    """
    Must be called explicitly at app startup (not on import).
    Loads FAISS index from FAISS_INDEX_PATH.
    Loads Parquet DataFrame from PARQUET_PATH.
    Stores both in module-level variables.
    """

def retrieve(user_query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Embeds user_query, runs FAISS search, returns top_k matching rows
    from the Parquet DataFrame. Columns match argo_profiles schema.
    """
```

---

### `scripts/build_index.py`

**Single job:** Run the indexer as a one-command script.

```python
# Must re-run every time new data is ingested
from rag import indexer
indexer.build_and_save()   # or equivalent call into indexer.py
```

### ✅ Verification

```
python scripts/build_index.py
# Expected: "Indexed N vectors → saved to indexes/argo.faiss"

# Quick smoke test in Python REPL:
from rag.retriever import load_index, retrieve
load_index()
result = retrieve("temperature at 500m in Indian Ocean")
assert len(result) == 5
print(result.columns.tolist())   # must match argo_profiles columns
```

---

## Phase 3 — LLM Integration

**Goal:** Combine retrieved context with GPT-4 to produce grounded text answers and SQL.

---

### `llm/prompts.py`

**Single job:** Define all prompt templates as string constants. No logic here.

**Required templates:**

```python
CHAT_PROMPT = """
You are SeaBorg, an expert ocean data analyst. Answer the user's question using
ONLY the data records provided below. Be specific, cite float IDs and dates.
If the data does not support the question, say so clearly.

Context records:
{context}

Question: {question}

Answer:
"""

SQL_PROMPT = """
Convert the following question into a valid PostgreSQL SELECT query for the
table `argo_profiles` with columns:
id, float_id, date, latitude, longitude, depth_m, temp_c, salinity, oxygen, created_at.

Return ONLY the SQL query. No explanation. No markdown. No semicolon at the end.

Question: {question}
"""
```

**Required function:**
```python
def build_prompt(question: str, context_rows: pd.DataFrame) -> str:
    """
    Formats context_rows as a bullet list and fills CHAT_PROMPT.
    Each bullet: "• Float {float_id} | {date} | {depth_m}m | {temp_c}°C | {salinity} PSU"
    """
```

---

### `llm/query_engine.py`

**Single job:** Run the full RAG + LLM call; return answer and generated SQL.

**Rules:**
- Read model name from env var `LLM_MODEL`, default `"gpt-4"`. Never hardcode.
- Use temperature `0.2` or lower for factual reproducibility.

**Public interface:**
```python
def answer_query(question: str, context_rows: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (answer_text, sql_string).
    Builds prompt via prompts.build_prompt(), calls OpenAI, returns response content.
    Also calls nl_to_sql.generate_sql(question) to populate sql_string.
    """
```

---

### `llm/nl_to_sql.py`

**Single job:** Translate a natural language question to a safe SQL query and execute it.

**CRITICAL — forbidden SQL keywords (case-insensitive):**
`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `GRANT`

If any forbidden keyword appears in the generated SQL: return `(None, "Unsafe SQL rejected")` without executing.

**Public interface:**
```python
def generate_sql(question: str) -> str:
    """Sends SQL_PROMPT to LLM. Returns raw SQL string (may be unsafe — validate before use)."""

def safe_sql_query(sql: str, engine: Engine) -> tuple[pd.DataFrame | None, str | None]:
    """
    Validates sql against forbidden keywords.
    If safe: executes via SQLAlchemy, returns (DataFrame, None).
    If unsafe: returns (None, error_message).
    """
```

### ✅ Verification

```python
from rag.retriever import load_index, retrieve
from llm.query_engine import answer_query
load_index()
rows = retrieve("what is the average temperature at 200m depth?")
answer, sql = answer_query("what is the average temperature at 200m depth?", rows)
print(answer)   # must be a non-empty string referencing specific float data
print(sql)      # must be a SELECT statement
```

---

## Phase 4 — FastAPI Backend

**Goal:** Expose all functionality as a documented REST API.

---

### `api/models.py`

**Single job:** Pydantic v2 request and response schemas. No logic.

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    chart_type: Literal["map", "profile", "timeseries", "none"]
    float_ids: list[str]
    sql_used: str
    confidence: float  # 0.0 to 1.0

class FloatDataRequest(BaseModel):
    float_id: str
    start_date: str | None = None
    end_date: str | None = None
    depth_min: float | None = None
    depth_max: float | None = None

class ExportRequest(BaseModel):
    float_ids: list[str]
    format: Literal["csv", "netcdf"]
    start_date: str | None = None
    end_date: str | None = None
```

---

### `api/tools.py`

**Single job:** Define OpenAI tool-call specifications for LLM function-calling. No logic.

Define these three tools as Python dicts following the OpenAI `tools` array format:

**`get_float_data`**
- Description: Retrieve all readings for a specific ARGO float within a date range.
- Parameters: `float_id` (string, required), `start_date` (string, optional), `end_date` (string, optional)

**`find_nearby_floats`**
- Description: Find all floats within a given radius of a lat/lon coordinate.
- Parameters: `latitude` (number, required), `longitude` (number, required), `radius_km` (number, required)

**`get_parameter_stats`**
- Description: Return summary statistics (mean, min, max, std) for a parameter across a region.
- Parameters: `parameter` (enum: `temp_c`, `salinity`, `oxygen`), `region_name` (string), `depth_min` (number, optional), `depth_max` (number, optional)

---

### `api/routes/chat.py`

**Single job:** Handle `POST /chat`; return `ChatResponse`.

**`detect_chart_type(message: str) -> str` — keyword classifier (exact rules):**

| Return value | Trigger keywords |
|---|---|
| `"map"` | where, map, location, region, ocean, sea, coordinates |
| `"profile"` | depth, profile, pressure, meter, vertical |
| `"timeseries"` | trend, over time, monthly, year, history, change |
| `"none"` | (none of the above matched) |

Evaluate in this order. First match wins.

**`POST /chat` handler logic:**
```
1. rows = retrieve(request.message, top_k=5)
2. answer, sql = answer_query(request.message, rows)
3. chart_type = detect_chart_type(request.message)
4. float_ids = rows["float_id"].unique().tolist()
5. return ChatResponse(answer=answer, chart_type=chart_type,
                        float_ids=float_ids, sql_used=sql, confidence=0.85)
```

---

### `api/routes/data.py`

**Single job:** Read-only data query endpoints.

| Method | Path | Description |
|---|---|---|
| `GET` | `/floats` | Paginated list of unique float IDs with date range and bounding box |
| `GET` | `/float/{float_id}` | All readings for one float; optional query params: `start_date`, `end_date`, `depth_min`, `depth_max` |
| `GET` | `/stats` | Aggregate stats: total row count, date range, geographic coverage |

---

### `api/routes/export.py`

**Single job:** Handle `POST /export`; stream a file download.

- CSV format: `StreamingResponse` with `Content-Type: text/csv`, `Content-Disposition: attachment; filename="seaborg_export.csv"`
- NetCDF format: `StreamingResponse` with `Content-Type: application/octet-stream`, `Content-Disposition: attachment; filename="seaborg_export.nc"`
- Query the database using `ExportRequest` fields as filters before writing the file.

---

### `api/main.py`

**Single job:** Assemble app, configure CORS, register routers, run startup events.

**Startup event must:**
1. Call `retriever.load_index()` — FAISS must be in memory before first request.
2. Verify database connection with a simple `SELECT 1` query.
3. Print `"SeaBorg API ready."` on success.

**CORS rules:**
- `ENVIRONMENT=development` → allow all origins, methods, headers.
- `ENVIRONMENT=production` → restrict `allow_origins` to the deployed frontend URL only.

**Router registration:** All three routers mounted under prefix `/api`.

### ✅ Verification

```bash
uvicorn api.main:app --reload
# Expected console output: "SeaBorg API ready."
# Open browser: http://localhost:8000/docs — must load Swagger UI

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the temperature at 500m in the Indian Ocean?"}'
# Expected: JSON with answer, chart_type, float_ids, sql_used, confidence
```

---

## Phase 5 — Visualisation Engine

**Goal:** Three Plotly chart functions, each accepting a DataFrame and returning a `go.Figure`.

---

### `visualisation/map_chart.py`

**Single job:** Geospatial scatter map of float positions coloured by temperature.

**Required inputs:** DataFrame with columns `latitude`, `longitude`, `temp_c`, `float_id`, `date`, `depth_m`, `salinity`

**Required chart properties:**
- Chart type: `plotly.express.scatter_geo` or `go.Scattergeo`
- Color scale: `RdBu_r` (red = warm, blue = cold), mapped to `temp_c`
- Hover tooltip: float ID, date, depth, temperature, salinity
- Map style: show land, light blue ocean, country borders
- Title: `"ARGO Float Positions"`

```python
def plot_float_map(df: pd.DataFrame) -> go.Figure: ...
```

---

### `visualisation/profile_chart.py`

**Single job:** Depth-vs-variable profile chart for a single float.

**Required chart properties:**
- Filter to `float_id`, sort by `depth_m` ascending
- Y-axis: `depth_m` — **must be inverted** (surface = top, deep = bottom)
- X-axis: the selected variable
- Render both a line and scatter markers
- Label axes with units (e.g. "Temperature (°C)", "Depth (m)")
- Title: `"{variable} Profile — Float {float_id}"`

```python
def plot_depth_profile(df: pd.DataFrame, float_id: str, variable: str = "temp_c") -> go.Figure: ...
```

---

### `visualisation/timeseries_chart.py`

**Single job:** Time series of daily or monthly averages for one float.

**Required chart properties:**
- Filter to `float_id`, group by date, compute mean of variable
- If more than 90 data points → aggregate to monthly averages before plotting
- X-axis: date, Y-axis: variable mean
- Render as a line chart
- Title: `"{variable} over Time — Float {float_id}"`

```python
def plot_timeseries(df: pd.DataFrame, float_id: str, variable: str = "temp_c") -> go.Figure: ...
```

---

### `visualisation/exporter.py`

**Single job:** Save data and chart files to `data/exports/`.

```python
def export_csv(df: pd.DataFrame, filename: str) -> str:
    """Saves df as CSV to data/exports/. Returns full file path."""

def export_chart_html(fig: go.Figure, filename: str) -> str:
    """Saves interactive Plotly figure as .html. Returns full file path."""

def export_chart_png(fig: go.Figure, filename: str) -> str:
    """Saves Plotly figure as .png. Returns full file path."""
```

### ✅ Verification

```python
import pandas as pd
from visualisation.map_chart import plot_float_map
from visualisation.profile_chart import plot_depth_profile
from visualisation.timeseries_chart import plot_timeseries

df = pd.read_parquet("data/processed/argo.parquet")
fig1 = plot_float_map(df.head(50))
fig2 = plot_depth_profile(df, df["float_id"].iloc[0])
fig3 = plot_timeseries(df, df["float_id"].iloc[0])

assert fig1 is not None and fig2 is not None and fig3 is not None
print("All charts: OK")
```

---

## Phase 6 — Streamlit Frontend

**Goal:** A two-column chat + chart interface runnable with `streamlit run frontend/app.py`.

---

### `frontend/app.py`

**Single job:** Assemble layout, manage session state, call the API.

**Page config:**
```python
st.set_page_config(page_title="SeaBorg", page_icon="🌊", layout="wide")
```

**Session state keys to initialise if absent:**
```python
st.session_state.messages      # list of {"role": "user"|"assistant", "content": str}
st.session_state.last_response # last ChatResponse dict or None
st.session_state.filters       # dict from sidebar.render_sidebar()
```

**Layout:**
```python
sidebar_filters = sidebar.render_sidebar()
col_chat, col_chart = st.columns(2)
with col_chat:
    user_input = chat_panel.render_chat()
with col_chart:
    chart_panel.render_chart()
```

**On user input (non-None):**
```python
# 1. Append user message to history
# 2. Show spinner: "Searching ARGO data..."
# 3. POST to {API_URL}/api/chat with {"message": user_input}
# 4. On success: append assistant answer to history, store full response
# 5. On connection error: append error message to history (no traceback shown to user)
# 6. st.rerun()
```

---

### `frontend/components/chat_panel.py`

**Single job:** Render message history and chat input; return user input string or None.

```python
def render_chat() -> str | None:
    """
    Iterates st.session_state.messages.
    user role → st.chat_message("user")  with avatar "🧑"
    assistant role → st.chat_message("assistant") with avatar "🌊"
    Input box via st.chat_input("Ask about ocean data...")
    Returns the submitted string or None.
    """
```

---

### `frontend/components/chart_panel.py`

**Single job:** Read `st.session_state.last_response` and render the appropriate chart.

**Logic:**
```python
def render_chart() -> None:
    resp = st.session_state.get("last_response")
    if resp is None:
        st.info("Ask a question to see a chart here.")
        return

    df = pd.read_parquet(os.getenv("PARQUET_PATH"))
    float_ids = resp["float_ids"]
    chart_df = df[df["float_id"].isin(float_ids)]

    chart_type = resp["chart_type"]
    if chart_type == "map":
        fig = plot_float_map(chart_df)
    elif chart_type == "profile":
        fig = plot_depth_profile(chart_df, float_ids[0])
    elif chart_type == "timeseries":
        fig = plot_timeseries(chart_df, float_ids[0])
    else:
        st.info("No visualisation for this query.")
        return

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("SQL used"):
        st.code(resp["sql_used"], language="sql")

    csv = chart_df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "seaborg_export.csv", "text/csv")
```

---

### `frontend/components/sidebar.py`

**Single job:** Render filter controls; return selected values as a dict.

**Required controls:**

| Control | Widget | Options / Range |
|---|---|---|
| Ocean regions | `st.multiselect` | Indian Ocean, Atlantic, Pacific, Southern, Arctic |
| Date range | `st.date_input` (range) | Default: full available range from DB |
| Depth range | `st.slider` (range) | 0 to 2000 m |
| Variable | `st.selectbox` | Temperature, Salinity, Oxygen |

**Return type:**
```python
{
    "regions":    list[str],
    "start_date": datetime.date,
    "end_date":   datetime.date,
    "depth_min":  int,
    "depth_max":  int,
    "variable":   str   # "temp_c" | "salinity" | "oxygen"
}
```

### ✅ Verification

```bash
streamlit run frontend/app.py
# Must open browser page with two columns (chat left, chart right) and sidebar
# Type: "Show me temperature data in the Indian Ocean"
# Expected: assistant reply appears + map chart renders on right
```

---

## Phase 7 — Tests

**Goal:** All tests pass with `pytest tests/ -v`.

---

### `tests/test_ingestion.py`

Write tests that verify:
1. `parser.parse_netcdf(sample_nc)` returns a DataFrame containing all required columns: `float_id`, `date`, `latitude`, `longitude`, `depth_m`, `temp_c`, `salinity`.
2. `qc_filter.apply_qc(df, ds)` removes rows where `TEMP_QC != 1` or `PSAL_QC != 1`.
3. `qc_filter.apply_qc(df, ds)` removes rows with `temp_c < -3` or `temp_c > 40`.
4. `qc_filter.apply_qc(df, ds)` removes rows with `salinity < 20` or `salinity > 42`.
5. Calling `db_loader.save_to_postgres(df)` twice does not duplicate rows (unique constraint test or row count check).

---

### `tests/test_rag.py`

Write tests that verify:
1. `summariser.summarise_row(row)` returns a non-empty string that contains the float ID and temperature value from the row.
2. `embedder.embed_query("test query")` returns a numpy array with shape `(1, 384)`.
3. `embedder.embed_texts(["text1", "text2"])` returns a numpy array with shape `(2, 384)`.
4. After `load_index()`, `retrieve("temperature in Indian Ocean")` returns a DataFrame with exactly 5 rows.
5. The returned DataFrame contains all columns of `argo_profiles`.

---

### `tests/test_api.py`

Use FastAPI's `TestClient` (synchronous). Write tests that verify:
1. `POST /api/chat` with `{"message": "What is the ocean temperature?"}` returns HTTP 200.
2. The response body contains keys: `answer`, `chart_type`, `float_ids`, `sql_used`, `confidence`.
3. `chart_type` value is one of `["map", "profile", "timeseries", "none"]`.
4. A message containing `"DROP TABLE"` — the `sql_used` field in the response must NOT contain `DROP TABLE` (the safety filter must have caught it).
5. `GET /api/floats` returns HTTP 200 with a non-empty JSON list.

---

### `tests/test_charts.py`

Write tests that verify:
1. `plot_float_map(df)` returns a `plotly.graph_objects.Figure` instance.
2. `plot_float_map(df)` figure has at least one trace.
3. `plot_depth_profile(df, float_id)` returns a `go.Figure` instance.
4. `plot_depth_profile(df, float_id)` figure Y-axis has `autorange="reversed"` (inverted).
5. `plot_timeseries(df, float_id)` returns a `go.Figure` instance.
6. `exporter.export_csv(df, "test")` creates a file at the expected path.

---

## Phase 8 — Deployment

### Backend — Render.com

| Field | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Environment vars | Copy all keys from `.env`; set `ENVIRONMENT=production` |
| Database | Provision a Render managed PostgreSQL; use its connection string as `DATABASE_URL` |

### Frontend — Streamlit Community Cloud

| Field | Value |
|---|---|
| Repository | Your GitHub repo |
| Main file | `frontend/app.py` |
| Secrets | `OPENAI_API_KEY`, `DATABASE_URL`, `API_URL` (set to Render backend's public HTTPS URL) |

### Pre-deployment Checklist

- [ ] `pytest tests/ -v` → zero failures
- [ ] `.env` is in `.gitignore` and not tracked
- [ ] `requirements.txt` is up to date (`pip freeze > requirements.txt`)
- [ ] FAISS index + Parquet file are either committed or `build_index.py` runs on first boot
- [ ] `api/main.py` CORS `allow_origins` restricted to production frontend URL
- [ ] All env vars set in Render and Streamlit dashboards
- [ ] `POST /api/chat` returns valid response on the live URL

### ✅ Final Verification

```
# All of these must be true simultaneously for the project to be complete:

python scripts/run_ingestion.py        → completes without errors
python scripts/build_index.py         → prints vector count > 0
uvicorn api.main:app --reload         → starts, prints "SeaBorg API ready."
curl POST /api/chat {real question}   → returns valid ChatResponse JSON
streamlit run frontend/app.py         → working chatbot + live charts in browser
pytest tests/ -v                      → all tests pass, zero failures
https://{your-deployed-url}           → accessible in browser over HTTPS
```

---

## 16. Data Flow Contracts

These are the exact function signatures modules must honour. Changing these breaks
the integration between phases.

```
parser.parse_netcdf(filepath: str)
    → (pd.DataFrame, xr.Dataset)

qc_filter.apply_qc(df: pd.DataFrame, dataset: xr.Dataset)
    → pd.DataFrame

db_loader.save_to_postgres(df: pd.DataFrame) → None
db_loader.save_to_parquet(df: pd.DataFrame)  → None

summariser.summarise_row(row: dict | pd.Series) → str

embedder.embed_texts(texts: list[str])  → np.ndarray  # shape (n, 384), float32
embedder.embed_query(query: str)        → np.ndarray  # shape (1, 384), float32

retriever.load_index()                              → None
retriever.retrieve(query: str, top_k: int = 5)     → pd.DataFrame

prompts.build_prompt(question: str, context_rows: pd.DataFrame) → str

query_engine.answer_query(question: str, context_rows: pd.DataFrame)
    → (str, str)   # (answer_text, sql_string)

nl_to_sql.generate_sql(question: str) → str
nl_to_sql.safe_sql_query(sql: str, engine: Engine)
    → (pd.DataFrame | None, str | None)

plot_float_map(df: pd.DataFrame)                              → go.Figure
plot_depth_profile(df: pd.DataFrame, float_id: str,
                   variable: str = "temp_c")                  → go.Figure
plot_timeseries(df: pd.DataFrame, float_id: str,
                variable: str = "temp_c")                     → go.Figure
```

---

## 17. Global Rules

These rules apply to every line of code in the project. No exceptions.

### Environment & configuration
- Every module that reads env vars must call `load_dotenv()` at the top.
- Use `os.getenv("VAR_NAME")` everywhere. No hardcoded values.
- The model name `LLM_MODEL` is read from env — never written in source.

### Imports
- Use relative imports within packages: `from .embedder import embed_query` not `from rag.embedder import embed_query`.
- Exception: scripts in `scripts/` use absolute imports.

### Documentation
- Every public function must have a docstring that states: what it accepts, what it returns, and any side effects.

### Safety
- Never execute LLM-generated SQL without calling `safe_sql_query()` first.
- Never commit `.env`.

### Performance
- The FAISS index and Parquet DataFrame are loaded once at startup. Never reload per request.
- The embedding model is loaded lazily on first use. Never on import.

### Error handling
- The Streamlit frontend must never show a Python traceback to the user. Catch exceptions and display human-readable messages.

### Build discipline
- Verify each phase passes its ✅ Verification check before starting the next phase.
- If any test or script fails, fix it before proceeding.
- Re-run `scripts/build_index.py` every time new data is ingested.
