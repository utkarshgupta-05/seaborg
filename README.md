# SeaBorg — AI-Powered Ocean Data Chatbot

> Query, explore, and visualise ARGO float oceanographic data using plain natural language.

**Team:**
- Shubham Kumar — 24BCS237
- Shubham Kulkarni — 24BCS236
- Pushkar Trivedi — 24BCS203
- Nishkarsh Sharma — 24BCS176
- Utkarsh Gupta — 24BCS256

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Tech Stack](#3-tech-stack)
4. [Environment Setup](#4-environment-setup)
5. [Phase 1 — Data Ingestion & ETL](#5-phase-1--data-ingestion--etl)
6. [Phase 2 — RAG Pipeline & Vector Database](#6-phase-2--rag-pipeline--vector-database)
7. [Phase 3 — FastAPI Backend](#7-phase-3--fastapi-backend)
8. [Phase 4 — Visualisation Engine](#8-phase-4--visualisation-engine)
9. [Phase 5 — React Frontend](#9-phase-5--react-frontend)
10. [Phase 6 — Testing & Deployment](#10-phase-6--testing--deployment)
11. [Agent Build Instructions](#11-agent-build-instructions)

## 1. Project Overview

SeaBorg is an AI-powered conversational web app that makes ARGO float oceanographic data
accessible to researchers, students, and policymakers through natural language queries.

The system uses a FastAPI backend, a React + Vite frontend, a Groq-hosted LLM, and a RAG
pipeline grounded in real ARGO measurements stored in PostgreSQL and Parquet.

**What ARGO data is:** ARGO floats are robotic ocean sensors that sink and rise in the sea,
recording temperature, salinity, and pressure at different depths. The data is stored in
NetCDF files — a multi-dimensional scientific format.

**What SeaBorg does:**
- Accepts plain English questions ("Show me temperature at 500m in the Indian Ocean")
- Searches ARGO data using a RAG (Retrieval-Augmented Generation) pipeline
- Returns an AI-generated answer + an interactive chart (map, profile, or time series)
- Allows export of results as CSV or NetCDF

**Business model:** Open-source core, freemium for advanced analytics and enterprise support.

## 2. Folder Structure

Create this exact structure before writing any code.
```
seaborg/
├── README.md
├── requirements.txt
├── .env                          # API keys — NEVER commit to git
├── .env.example                  # Safe template to commit
├── .gitignore
│
├── data/
│   ├── raw/                      # Original .nc files downloaded from ARGO
│   ├── processed/                # Cleaned Parquet files after ETL
│   └── exports/                  # CSV/NetCDF files users download
│
├── ingestion/                    # PHASE 1
│   ├── __init__.py
│   ├── downloader.py             # Download .nc files from EuroArgo
│   ├── parser.py                 # Parse NetCDF with xarray
│   ├── qc_filter.py              # Remove bad quality-control flag readings
│   └── db_loader.py              # Insert cleaned rows into PostgreSQL
│
├── rag/                          # PHASE 2
│   ├── __init__.py
│   ├── summariser.py             # Convert data rows to text summaries
│   ├── embedder.py               # Generate vector embeddings
│   ├── indexer.py                # Build and save FAISS index
│   └── retriever.py              # Search FAISS, return top-k rows
│
├── llm/                          # PHASE 2 + 3
│   ├── __init__.py
│   ├── prompts.py                # Prompt templates
│   ├── query_engine.py           # RAG + LLM orchestration
│   └── nl_to_sql.py              # Natural language to SQL translation
│
├── api/                          # PHASE 3
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── models.py                 # Pydantic request/response schemas
│   ├── tools.py                  # Tool definitions for function calling
│   └── routes/
│       ├── __init__.py
│       ├── chat.py               # POST /chat endpoint
│       ├── data.py               # GET /floats, /float/{id} endpoints
│       └── export.py             # POST /export endpoint
│
├── visualisation/                # PHASE 4
│   ├── __init__.py
│   ├── map_chart.py              # Geospatial float position map
│   ├── profile_chart.py          # Depth vs temperature/salinity profile
│   ├── timeseries_chart.py       # Variable trend over time
│   └── exporter.py               # CSV, PNG, HTML export helpers
│
├── frontend/                     # PHASE 5
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api/
│       │   └── client.js
│       ├── components/
│       │   ├── ChatPanel.jsx
│       │   ├── ChartPanel.jsx
│       │   ├── Sidebar.jsx
│       │   └── MessageBubble.jsx
│       ├── pages/
│       │   └── Dashboard.jsx
│       ├── hooks/
│       │   └── useChat.js
│       └── styles/
│           └── globals.css
│
├── tests/                        # PHASE 6
│   ├── test_ingestion.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_charts.py
│
├── notebooks/                    # Jupyter exploration (optional but helpful)
│   ├── 01_explore_netcdf.ipynb
│   ├── 02_test_rag.ipynb
│   └── 03_visualisation_demo.ipynb
│
├── scripts/
│   ├── setup_db.py               # Create PostgreSQL tables (run once)
│   ├── run_ingestion.py          # Run full ETL pipeline
│   └── build_index.py            # Build FAISS index from processed data
│
├── indexes/
│   └── argo_index.faiss          # Saved FAISS vector index
│
└── models/
    └── .gitkeep                  # Cached embedding models go here
```

Create all folders at once:
```bash
mkdir -p seaborg/{data/{raw,processed,exports},ingestion,rag,llm,api/routes,visualisation,frontend/src/{api,components,pages,hooks,styles},tests,notebooks,scripts,indexes,models}
touch seaborg/ingestion/__init__.py seaborg/rag/__init__.py seaborg/llm/__init__.py seaborg/api/__init__.py seaborg/api/routes/__init__.py seaborg/visualisation/__init__.py
```

## 3. Tech Stack

| Category | Library / Tool | Notes |
|---|---|---|
| NetCDF parsing | `xarray`, `netCDF4` | Open `.nc` files |
| Tabular data | `pandas`, `numpy` | All DataFrame ops |
| Parquet I/O | `pyarrow` | Read/write Parquet |
| Database | `PostgreSQL` + `SQLAlchemy` + `psycopg2-binary` | ORM layer |
| Embeddings | `sentence-transformers` model `all-MiniLM-L6-v2` | 384-dim, local, no GPU |
| Vector search | `faiss-cpu` | Keep the implementation FAISS-first; do not introduce an alternate vector store unless the project explicitly requires it |
| LLM client | `groq` Python library | Groq-hosted chat completions |
| Backend | `FastAPI` + `uvicorn` | ASGI server |
| Schemas | `pydantic` v2 | All request/response models |
| Charts | `plotly` | All three chart types; no matplotlib |
| Frontend | `React` + `Vite` | Browser UI for chat and charts |
| Testing | `pytest` + `httpx` | `httpx` for async API tests |
| Ocean data | `argopy` | Helpful ARGO data access utilities |

## 4. Environment Setup

### Step 1 — Python virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Step 2 — Node.js frontend environment
Install Node.js 18+ and then set up the React app:
```bash
cd frontend
npm install
```

### Step 3 — PostgreSQL
```bash
# Mac
brew install postgresql@14
brew services start postgresql@14
createdb seaborg

# Ubuntu/Linux
sudo apt install postgresql
sudo systemctl start postgresql
sudo -u postgres createdb seaborg

# Windows: download installer from https://www.postgresql.org/download/
```

### Step 4 — .env files

Create `.env` in the project root (never commit this):
```env
DATABASE_URL=postgresql://user:password@localhost/seaborg
GROQ_API_KEY=gsk-...
GROQ_MODEL=llama-3.3-70b-versatile
ENVIRONMENT=development
FAISS_INDEX_PATH=indexes/argo_index.faiss
PARQUET_PATH=data/processed/argo_profiles.parquet
API_URL=http://localhost:8000
```

Create `frontend/.env.local` for the React app:
```env
VITE_API_URL=http://localhost:8000
```

Create `.env.example` to commit:
```env
DATABASE_URL=postgresql://user:password@localhost/seaborg
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
ENVIRONMENT=development
FAISS_INDEX_PATH=indexes/argo_index.faiss
PARQUET_PATH=data/processed/argo_profiles.parquet
API_URL=http://localhost:8000
```

Add to `.gitignore`:
```text
.env
frontend/.env.local
venv/
__pycache__/
*.pyc
indexes/
models/
data/raw/
data/processed/
data/exports/
node_modules/
frontend/dist/
```

## 5. Phase 1 — Data Ingestion & ETL

**Goal:** Download ARGO `.nc` files, parse them, remove bad readings, and store clean data
in PostgreSQL and Parquet format.

**Deliverable:** `python scripts/run_ingestion.py` prints row counts and confirms DB insert.

---

### scripts/setup_db.py

Run this once before anything else to create your database table.
```python
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS argo_profiles (
            id          SERIAL PRIMARY KEY,
            float_id    VARCHAR(20),
            date        TIMESTAMP,
            latitude    FLOAT,
            longitude   FLOAT,
            depth_m     FLOAT,
            temp_c      FLOAT,
            salinity    FLOAT,
            oxygen      FLOAT,
            created_at  TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_float_id ON argo_profiles(float_id);
        CREATE INDEX IF NOT EXISTS idx_date     ON argo_profiles(date);
        CREATE INDEX IF NOT EXISTS idx_latlon   ON argo_profiles(latitude, longitude);
    """))
    conn.commit()

print("Database tables created successfully.")
```

Run it:
```bash
python scripts/setup_db.py
```

---

### ingestion/parser.py
```python
import xarray as xr
import pandas as pd

def parse_nc_file(filepath: str) -> pd.DataFrame:
    ds = xr.open_dataset(filepath, decode_times=True)

    vars_needed = ["PRES", "TEMP", "PSAL", "LATITUDE", "LONGITUDE", "JULD"]
    df = ds[vars_needed].to_dataframe().reset_index()

    df = df.rename(columns={
        "PRES":      "depth_m",
        "TEMP":      "temp_c",
        "PSAL":      "salinity",
        "LATITUDE":  "latitude",
        "LONGITUDE": "longitude",
        "JULD":      "date",
    })

    float_id = str(filepath).split("/")[-1].split("_")[0]
    df["float_id"] = float_id

    return df.dropna(subset=["temp_c", "salinity", "depth_m"]), ds
```

---

### ingestion/qc_filter.py
```python
import pandas as pd

def apply_qc_filter(df: pd.DataFrame, ds) -> pd.DataFrame:
    # QC flag 1 = good data, 4 = bad — keep only good readings
    if "TEMP_QC" in ds:
        temp_qc = ds["TEMP_QC"].values.flatten().astype(str)
        df = df[temp_qc == "1"]
    if "PSAL_QC" in ds:
        psal_qc = ds["PSAL_QC"].values.flatten().astype(str)
        df = df[psal_qc == "1"]

    # Sanity range checks
    df = df[(df["temp_c"] > -3)  & (df["temp_c"] < 40)]
    df = df[(df["salinity"] > 20) & (df["salinity"] < 42)]
    df = df[df["depth_m"] > 0]

    return df
```

---

### ingestion/db_loader.py
```python
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def load_to_db(df: pd.DataFrame):
    engine = create_engine(os.getenv("DATABASE_URL"))
    df.to_sql("argo_profiles", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows to PostgreSQL.")

def save_parquet(df: pd.DataFrame, path: str = "data/processed/argo_profiles.parquet"):
    df.to_parquet(path, index=False)
    print(f"Saved Parquet: {path}")
```

---

### scripts/run_ingestion.py
```python
import os
from ingestion.parser import parse_nc_file
from ingestion.qc_filter import apply_qc_filter
from ingestion.db_loader import load_to_db, save_parquet

RAW_DIR = "data/raw"

all_frames = []

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".nc"):
        filepath = os.path.join(RAW_DIR, filename)
        print(f"Parsing {filename}...")
        df, ds = parse_nc_file(filepath)
        print(f"  Raw rows: {len(df)}")
        df = apply_qc_filter(df, ds)
        print(f"  After QC: {len(df)} rows")
        all_frames.append(df)

import pandas as pd
combined = pd.concat(all_frames, ignore_index=True)
load_to_db(combined)
save_parquet(combined)
print(f"\nDone. Total rows stored: {len(combined)}")
```

---

## 6. Phase 2 — RAG Pipeline & Vector Database

**Goal:** Convert database rows into searchable text embeddings. At query time, find the
most relevant ARGO records and pass them as context to the LLM.

**Deliverable:** `retrieve("temperature 500m Indian Ocean")` returns a matching DataFrame.

**How RAG works (plain English):**
1. Offline: convert each data row into a short text summary, then into a vector (list of
   numbers capturing meaning). Store all vectors in FAISS.
2. At query time: convert the user's question into a vector using the same model, search
   FAISS for the 5 most similar vectors, return those rows as context.
3. Pass context + question into the LLM prompt to get a data-grounded answer.

---

### rag/summariser.py
```python
def row_to_summary(row: dict) -> str:
    return (
        f"Float {row['float_id']} measured on {row['date']:%Y-%m-%d}. "
        f"Location: latitude {row['latitude']:.2f}, longitude {row['longitude']:.2f}. "
        f"At depth {row['depth_m']:.0f}m: temperature {row['temp_c']:.1f}C, "
        f"salinity {row['salinity']:.2f} PSU."
    )
```

---

### rag/embedder.py
```python
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"   # free, runs locally, 384-dimensional vectors
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, show_progress_bar=True)

def embed_query(query: str) -> np.ndarray:
    return get_model().encode([query])
```

---

### scripts/build_index.py

Run this once after ingestion. Re-run it whenever new data is added.
```python
import faiss
import numpy as np
import pandas as pd
from rag.summariser import row_to_summary
from rag.embedder import embed_texts

df = pd.read_parquet("data/processed/argo_profiles.parquet")

print("Generating text summaries...")
summaries = [row_to_summary(row) for _, row in df.iterrows()]

print("Generating embeddings (this may take a few minutes)...")
embeddings = embed_texts(summaries).astype("float32")

print("Building FAISS index...")
dimension = embeddings.shape[1]   # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "indexes/argo_index.faiss")
print(f"Index saved. {index.ntotal} vectors indexed.")
```

---

### rag/retriever.py
```python
import faiss
import numpy as np
import pandas as pd
from rag.embedder import embed_query

index = None
df_ref = None

def load_index(
    index_path: str = "indexes/argo_index.faiss",
    parquet_path: str = "data/processed/argo_profiles.parquet"
):
    global index, df_ref
    index = faiss.read_index(index_path)
    df_ref = pd.read_parquet(parquet_path)
    print(f"FAISS index loaded: {index.ntotal} vectors.")

def retrieve(user_query: str, top_k: int = 5) -> pd.DataFrame:
    q_vec = embed_query(user_query).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    return df_ref.iloc[indices[0]]
```

---

### llm/prompts.py
```python
CHAT_PROMPT = """You are SeaBorg, an expert ocean data assistant.
Use ONLY the ARGO float data provided below to answer the question.
If the data does not contain the answer, say so clearly.
Be concise. Reference specific numbers from the data.

ARGO DATA:
{context}

USER QUESTION: {question}

ANSWER:"""

def build_prompt(question: str, context_rows) -> str:
    lines = []
    for _, r in context_rows.iterrows():
        lines.append(
            f"- Float {r.float_id} | {r.date:%Y-%m-%d} | "
            f"lat {r.latitude:.2f} lon {r.longitude:.2f} | "
            f"depth {r.depth_m:.0f}m | temp {r.temp_c:.1f}C | sal {r.salinity:.2f}"
        )
    context = "\n".join(lines)
    return CHAT_PROMPT.format(context=context, question=question)
```

---

### llm/query_engine.py
```python
from groq import Groq
import os
from llm.prompts import build_prompt
from rag.retriever import retrieve

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))

def answer_query(question: str, context_rows=None) -> tuple[str, str]:
    if context_rows is None:
        context_rows = retrieve(question)

    prompt = build_prompt(question, context_rows)
    sql_used = generate_sql(question)   # from nl_to_sql.py

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer, sql_used

def generate_sql(question: str) -> str:
    sql_prompt = f"""Convert this natural language question about ARGO ocean data into a
PostgreSQL query on the argo_profiles table (columns: float_id, date, latitude, longitude,
depth_m, temp_c, salinity, oxygen).

Question: {question}
Return ONLY the SQL, no explanation."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

---

### llm/nl_to_sql.py
```python
# SQL safety guard — always wrap LLM-generated SQL in this before executing
from sqlalchemy import text
import pandas as pd

FORBIDDEN = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT"]

def safe_sql_query(sql: str, engine) -> tuple[pd.DataFrame | None, str | None]:
    sql_upper = sql.upper()
    for word in FORBIDDEN:
        if word in sql_upper:
            return None, f"Blocked: query contains forbidden keyword '{word}'"
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(sql), conn)
        return result, None
    except Exception as e:
        return None, f"SQL error: {str(e)}"
```

---

## 7. Phase 3 — FastAPI Backend

**Goal:** Expose the RAG + LLM pipeline as a REST API. The frontend calls `POST /api/chat`
and receives a structured JSON response with the answer, chart type, and float IDs.

**Deliverable:** `uvicorn api.main:app --reload` starts with no errors. `/docs` page loads.

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
- Read model name from env vars `GROQ_MODEL` or `LLM_MODEL`, default `"llama-3.3-70b-versatile"`.
- Use the `groq` Python library for all LLM calls.
- Use temperature `0.2` or lower for factual reproducibility.

**Public interface:**
```python
def answer_query(question: str, context_rows: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (answer_text, sql_string).
    Builds prompt via prompts.build_prompt(), calls Groq, returns response content.
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
    """Sends SQL_PROMPT to Groq. Returns raw SQL string (may be unsafe — validate before use)."""

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

## 8. Phase 4 — Visualisation Engine

**Goal:** Build three chart generators triggered by `chart_type` in the API response.

**Deliverable:** Each function accepts a DataFrame and returns a Plotly figure.

---

### visualisation/map_chart.py
```python
import plotly.express as px
import pandas as pd

def plot_float_map(df: pd.DataFrame):
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="temp_c",
        color_continuous_scale="RdBu_r",
        hover_data=["float_id", "date", "depth_m", "salinity"],
        title="ARGO Float Positions",
    )
    fig.update_layout(geo=dict(
        showland=True,
        showocean=True,
        oceancolor="lightblue",
    ))
    return fig
```

---

### visualisation/profile_chart.py
```python
import plotly.graph_objects as go
import pandas as pd

def plot_depth_profile(df: pd.DataFrame, float_id: str, variable: str = "temp_c"):
    profile = df[df["float_id"] == float_id].sort_values("depth_m")
    label = "Temperature (°C)" if variable == "temp_c" else "Salinity (PSU)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile[variable],
        y=profile["depth_m"],
        mode="lines+markers",
        name=label,
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed", title="Depth (m)"),
        xaxis_title=label,
        title=f"{label} Profile — Float {float_id}",
    )
    return fig
```

---

### visualisation/timeseries_chart.py
```python
import plotly.express as px
import pandas as pd

def plot_timeseries(df: pd.DataFrame, float_id: str, variable: str = "temp_c"):
    ts = (
        df[df["float_id"] == float_id]
        .groupby("date")[variable]
        .mean()
        .reset_index()
    )
    label = "Temperature (°C)" if variable == "temp_c" else "Salinity (PSU)"
    fig = px.line(ts, x="date", y=variable, title=f"{label} over time — Float {float_id}")
    return fig
```

---

### visualisation/exporter.py
```python
import pandas as pd

def export_csv(df: pd.DataFrame, path: str = "data/exports/results.csv"):
    df.to_csv(path, index=False)
    return path

def export_chart_html(fig, path: str = "data/exports/chart.html"):
    fig.write_html(path)
    return path

def export_chart_png(fig, path: str = "data/exports/chart.png"):
    fig.write_image(path)
    return path
```

---

## 9. Phase 5 — React Frontend

**Goal:** A two-column React web app — chat on the left, live charts on the right.

**Deliverable:** `npm run dev` opens a working chatbot in the browser.

---

### `frontend/package.json`

**Core dependencies:**
```json
{
  "dependencies": {
    "axios": "^1.x",
    "plotly.js": "^2.x",
    "react-plotly.js": "^2.x",
    "react-router-dom": "^6.x"
  }
}
```

**Recommended dev dependencies:**
```json
{
  "devDependencies": {
    "tailwindcss": "^3.x",
    "postcss": "^8.x",
    "autoprefixer": "^10.x"
  }
}
```

---

### `frontend/src/api/client.js`

**Single job:** Create the API client used by all frontend components.

```javascript
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: { "Content-Type": "application/json" },
});
```

---

### `frontend/src/components/ChatPanel.jsx`

**Single job:** Render message history and chat input; return user input handling via props/hooks.

**Behaviour:**
- Use a modern chat UI with user and assistant message bubbles
- Keep conversation history in React state or a custom hook
- Submit new prompts to the backend through `POST /api/chat`
- Show a loading spinner while waiting for the response
- Never expose raw stack traces to the user

---

### `frontend/src/components/ChartPanel.jsx`

**Single job:** Read the latest chat response and render the appropriate Plotly chart.

**Behaviour:**
- Fetch data from the backend or use the returned float IDs to filter a local copy of the Parquet data
- Render `map`, `profile`, or `timeseries` charts using Plotly
- Display the SQL used in an expandable code block
- Provide a CSV download button for the current filtered dataset

---

### `frontend/src/components/Sidebar.jsx`

**Single job:** Render filter controls and return selected values to the dashboard.

**Recommended controls:**

| Control | Widget | Options / Range |
|---|---|---|
| Ocean regions | multi-select | Indian Ocean, Atlantic, Pacific, Southern, Arctic |
| Date range | date picker | Default: full available range |
| Depth range | slider | 0 to 2000 m |
| Variable | select | Temperature, Salinity, Oxygen |

---

### `frontend/src/pages/Dashboard.jsx`

**Single job:** Assemble the overall layout.

```jsx
// Layout idea:
// sidebar on the left
// chat panel center/left
// chart panel right
```

### `frontend/src/App.jsx`

**Single job:** Set up routing and top-level providers if needed.

---

### React frontend runtime rules

- Use `VITE_API_URL` to call the backend.
- Keep UI responsive for desktop and tablet widths.
- Use Plotly figures only; do not add legacy frontend files or Python-only UI runtime code.
- Keep downloaded data exports consistent with the backend schema.

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

### ✅ Verification

```bash
# Start backend in one terminal
uvicorn api.main:app --reload

# Start frontend in another terminal
cd frontend
npm install
npm run dev

# Must open browser page with chat + chart layout and sidebar
# Type: "Show me temperature data in the Indian Ocean"
# Expected: assistant reply appears + chart renders on right
```

## 10. Phase 6 — Testing & Deployment

**Goal:** All tests pass with `pytest tests/ -v`, and the app is deployed with a React frontend.

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

### Deployment — Backend on Render.com

| Field | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Environment vars | Copy all keys from `.env`; set `ENVIRONMENT=production` |
| Database | Provision a Render managed PostgreSQL; use its connection string as `DATABASE_URL` |

### Deployment — Frontend on Vercel

| Field | Value |
|---|---|
| Repository | Your GitHub repo |
| Framework preset | Vite |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment vars | `VITE_API_URL` set to the Render backend public HTTPS URL |

### Pre-deployment Checklist

- [ ] `pytest tests/ -v` → zero failures
- [ ] `.env` is in `.gitignore` and not tracked
- [ ] `requirements.txt` is up to date (`pip freeze > requirements.txt`)
- [ ] FAISS index + Parquet file are either committed or `build_index.py` runs on first boot
- [ ] `api/main.py` CORS `allow_origins` restricted to the deployed frontend URL
- [ ] All env vars set in Render and Vercel dashboards
- [ ] `POST /api/chat` returns valid response on the live URL
- [ ] React frontend loads successfully from the Vercel URL

### ✅ Final Verification

```
# All of these must be true simultaneously for the project to be complete:

python scripts/run_ingestion.py        → completes without errors
python scripts/build_index.py         → prints vector count > 0
uvicorn api.main:app --reload         → starts, prints "SeaBorg API ready."
curl POST /api/chat {real question}   → returns valid ChatResponse JSON
cd frontend && npm install && npm run dev
                                         → working chatbot + live charts in browser
pytest tests/ -v                      → all tests pass, zero failures
https://{your-deployed-backend-url}    → accessible in browser over HTTPS
https://{your-deployed-frontend-url}   → accessible in browser over HTTPS
```

## 11. Agent Build Instructions

If you are an AI coding agent using this README, follow these rules exactly:

1. **Create the full folder structure first** — run the `mkdir` command from Section 2 before writing any Python or React code.
2. **Create all `__init__.py` files** — every Python package folder under `ingestion/`, `rag/`, `llm/`, `api/`, and `visualisation/` needs one.
3. **Run `scripts/setup_db.py` before any ingestion** — fix all database connection errors before proceeding.
4. **Download at least 3 real `.nc` files** from `https://data-argo.ifremer.fr` into `data/raw/` before testing the parser.
5. **Complete each phase's deliverable check before starting the next phase** — do not write Phase 2 code until Phase 1 inserts rows successfully.
6. **Never hardcode API keys** — always use `os.getenv()` and load from `.env` with `python-dotenv`.
7. **Use `GROQ_API_KEY` and `GROQ_MODEL`** for all LLM calls — do not import or configure OpenAI.
8. **Re-run `scripts/build_index.py` every time new data is ingested** — the FAISS index must reflect current data.
9. **Wrap every LLM-generated SQL in `safe_sql_query()`** before executing against the database.
10. **Run `pytest tests/ -v` and fix all failures before deployment.**
11. **Never commit `.env`** — confirm it is listed in `.gitignore` before the first `git push`.
12. **Build and verify the React frontend** with `cd frontend && npm install && npm run dev` during development and `npm run build` before deployment.

### Quick-start command sequence (run in order)
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/setup_db.py

# Data
python scripts/run_ingestion.py
python scripts/build_index.py

# Run backend
uvicorn api.main:app --reload

# Run frontend
cd frontend
npm install
npm run dev
```

If all commands succeed and `http://localhost:5173` shows a chat interface, the build is complete.
