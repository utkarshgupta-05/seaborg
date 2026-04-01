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
9. [Phase 5 — Streamlit Frontend](#9-phase-5--streamlit-frontend)
10. [Phase 6 — Testing & Deployment](#10-phase-6--testing--deployment)
11. [Agent Build Instructions](#11-agent-build-instructions)

---

## 1. Project Overview

SeaBorg is an AI-powered conversational web app that makes ARGO float oceanographic data
accessible to researchers, students, and policymakers through natural language queries.

**What ARGO data is:** ARGO floats are robotic ocean sensors that sink and rise in the sea,
recording temperature, salinity, and pressure at different depths. The data is stored in
NetCDF files — a multi-dimensional scientific format.

**What SeaBorg does:**
- Accepts plain English questions ("Show me temperature at 500m in the Indian Ocean")
- Searches ARGO data using a RAG (Retrieval-Augmented Generation) pipeline
- Returns an AI-generated answer + an interactive chart (map, profile, or time series)
- Allows export of results as CSV or NetCDF

**Business model:** Open-source core, freemium for advanced analytics and enterprise support.

---

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
│   ├── tools.py                  # MCP tool definitions
│   └── routes/
│       ├── chat.py               # POST /chat endpoint
│       ├── data.py               # GET /floats, /profile endpoints
│       └── export.py             # GET /export endpoint
│
├── visualisation/                # PHASE 4
│   ├── __init__.py
│   ├── map_chart.py              # Geospatial float position map
│   ├── profile_chart.py          # Depth vs temperature/salinity profile
│   ├── timeseries_chart.py       # Variable trend over time
│   └── exporter.py               # CSV, PNG, HTML export helpers
│
├── frontend/                     # PHASE 5
│   ├── app.py                    # Main Streamlit entry point
│   └── components/
│       ├── chat_panel.py         # Chat messages UI component
│       ├── chart_panel.py        # Dynamic chart display component
│       └── sidebar.py            # Filters and settings sidebar
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
mkdir -p seaborg/{data/{raw,processed,exports},ingestion,rag,llm,api/routes,visualisation,frontend/components,tests,notebooks,scripts,indexes,models}
touch seaborg/ingestion/__init__.py seaborg/rag/__init__.py seaborg/llm/__init__.py seaborg/api/__init__.py seaborg/api/routes/__init__.py seaborg/visualisation/__init__.py seaborg/frontend/__init__.py
```

---

## 3. Tech Stack

| Layer | Tool / Library | Purpose |
|---|---|---|
| Data parsing | `xarray`, `netCDF4` | Read ARGO NetCDF files |
| Data frames | `pandas`, `numpy` | Manipulate and clean data |
| Storage (structured) | `PostgreSQL` | SQL-queryable rows per depth reading |
| Storage (analytical) | `pyarrow` (Parquet) | Fast columnar reads for embeddings |
| DB interface | `SQLAlchemy`, `psycopg2` | Python ↔ PostgreSQL |
| Embeddings | `sentence-transformers` | Convert text summaries to vectors |
| Vector search | `faiss-cpu` | Fast similarity search |
| Alt vector DB | `chromadb` | Simpler FAISS alternative |
| LLM | `openai` / `huggingface_hub` | GPT-4, Qwen, or LLaMA |
| Agent tools | MCP protocol | Let LLM call Python functions |
| Backend API | `fastapi`, `uvicorn` | REST API server |
| Validation | `pydantic` | Request/response schemas |
| Maps | `plotly`, `folium` | Interactive geospatial charts |
| UI framework | `streamlit` | Python-first web app |
| Testing | `pytest`, `httpx` | Unit and integration tests |
| Deployment | Render / Railway | Free cloud hosting |

### requirements.txt
```
# Data ingestion
xarray==2024.2.0
netCDF4==1.7.1
pandas==2.2.1
numpy==1.26.4
pyarrow==15.0.2

# Database
sqlalchemy==2.0.29
psycopg2-binary==2.9.9

# RAG & embeddings
sentence-transformers==2.7.0
faiss-cpu==1.8.0
chromadb==0.5.0

# LLM
openai==1.30.1
langchain==0.2.1

# Backend
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1

# Visualisation
plotly==5.22.0
folium==0.16.0

# Frontend
streamlit==1.35.0
requests==2.32.3

# Testing
pytest==8.2.0
httpx==0.27.0
```

---

## 4. Environment Setup

### Step 1 — Python virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Step 2 — PostgreSQL
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

### Step 3 — .env file

Create `.env` in the project root (never commit this):
```
DATABASE_URL=postgresql://user:password@localhost/seaborg
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
ENVIRONMENT=development
```

Create `.env.example` to commit:
```
DATABASE_URL=postgresql://user:password@localhost/seaborg
OPENAI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
ENVIRONMENT=development
```

Add to `.gitignore`:
```
.env
venv/
__pycache__/
*.pyc
indexes/
models/
data/raw/
data/processed/
```

---

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
import openai
import os
from llm.prompts import build_prompt
from rag.retriever import retrieve

openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_query(question: str, context_rows=None) -> tuple[str, str]:
    if context_rows is None:
        context_rows = retrieve(question)

    prompt = build_prompt(question, context_rows)
    sql_used = generate_sql(question)   # from nl_to_sql.py

    response = openai.chat.completions.create(
        model="gpt-4",
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

    response = openai.chat.completions.create(
        model="gpt-4",
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

### api/models.py
```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    chart_type: str        # "map" | "profile" | "timeseries" | "none"
    float_ids: list[str]
    sql_used: str
    confidence: float
```

---

### api/routes/chat.py
```python
from fastapi import APIRouter
from api.models import ChatRequest, ChatResponse
from llm.query_engine import answer_query
from rag.retriever import retrieve

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    results   = retrieve(req.message)
    answer, sql = answer_query(req.message, results)
    chart_type  = detect_chart_type(req.message)
    float_ids   = results["float_id"].unique().tolist()

    return ChatResponse(
        answer=answer,
        chart_type=chart_type,
        float_ids=float_ids,
        sql_used=sql,
        confidence=0.9,
    )

def detect_chart_type(msg: str) -> str:
    msg = msg.lower()
    if any(w in msg for w in ["map", "where", "location", "region"]):
        return "map"
    if any(w in msg for w in ["depth", "profile", "pressure", "meter"]):
        return "profile"
    if any(w in msg for w in ["trend", "over time", "monthly", "year"]):
        return "timeseries"
    return "none"
```

---

### api/main.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import chat, data, export
from rag.retriever import load_index

app = FastAPI(title="SeaBorg API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    load_index()
    print("SeaBorg API ready.")

app.include_router(chat.router,   prefix="/api")
app.include_router(data.router,   prefix="/api")
app.include_router(export.router, prefix="/api")
```

Start the server:
```bash
uvicorn api.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive API docs
```

Test with curl:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "temperature at 500m depth Indian Ocean"}'
```

---

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

## 9. Phase 5 — Streamlit Frontend

**Goal:** A two-column web app — chat on the left, live charts on the right.

**Deliverable:** `streamlit run frontend/app.py` opens a working chatbot in the browser.

---

### frontend/app.py
```python
import streamlit as st
import requests
import pandas as pd
from visualisation.map_chart import plot_float_map
from visualisation.profile_chart import plot_depth_profile
from visualisation.timeseries_chart import plot_timeseries

st.set_page_config(page_title="SeaBorg", layout="wide", page_icon="🌊")
st.title("SeaBorg — Ocean Data Chatbot")

API_URL = "http://localhost:8000/api/chat"
PARQUET  = "data/processed/argo_profiles.parquet"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

col_chat, col_chart = st.columns([1, 1])

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about ocean data...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Searching ARGO data..."):
            try:
                resp = requests.post(API_URL, json={"message": user_input}).json()
            except Exception as e:
                resp = {"answer": f"Error: {e}", "chart_type": "none", "float_ids": []}

        st.session_state.messages.append({"role": "assistant", "content": resp["answer"]})
        st.session_state.last_response = resp
        st.rerun()

with col_chart:
    st.subheader("Visualisation")
    r = st.session_state.last_response

    if r:
        df = pd.read_parquet(PARQUET)

        if r["chart_type"] == "map":
            st.plotly_chart(plot_float_map(df), use_container_width=True)

        elif r["chart_type"] == "profile" and r["float_ids"]:
            st.plotly_chart(
                plot_depth_profile(df, r["float_ids"][0]),
                use_container_width=True,
            )

        elif r["chart_type"] == "timeseries" and r["float_ids"]:
            st.plotly_chart(
                plot_timeseries(df, r["float_ids"][0]),
                use_container_width=True,
            )

        st.download_button(
            label="Download results as CSV",
            data=df.to_csv(index=False),
            file_name="argo_results.csv",
            mime="text/csv",
        )

    else:
        st.info("Ask a question on the left to see a chart here.")
```

Run it:
```bash
streamlit run frontend/app.py
# Opens automatically at http://localhost:8501
```

---

## 10. Phase 6 — Testing & Deployment

### tests/test_api.py
```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

TEST_QUESTIONS = [
    "temperature at 500m depth in Indian Ocean",
    "show me floats near Madagascar",
    "salinity trend over 2023",
]

def test_chat_returns_200():
    resp = client.post("/api/chat", json={"message": TEST_QUESTIONS[0]})
    assert resp.status_code == 200

def test_chat_response_shape():
    resp = client.post("/api/chat", json={"message": TEST_QUESTIONS[0]}).json()
    assert "answer" in resp
    assert resp["chart_type"] in ["map", "profile", "timeseries", "none"]
    assert isinstance(resp["float_ids"], list)

def test_sql_safety_guard():
    resp = client.post("/api/chat", json={"message": "DROP TABLE argo_profiles"})
    assert "DROP" not in resp.json().get("sql_used", "")

def test_empty_message():
    resp = client.post("/api/chat", json={"message": ""})
    assert resp.status_code in [200, 422]
```

Run tests:
```bash
pytest tests/ -v
```

### Deployment on Render.com (free, recommended)
```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Initial SeaBorg build"
git remote add origin https://github.com/YOUR_USERNAME/seaborg
git push -u origin main

# 2. Go to render.com
#    → New Web Service
#    → Connect your GitHub repo
#    → Build Command:  pip install -r requirements.txt
#    → Start Command:  uvicorn api.main:app --host 0.0.0.0 --port $PORT
#    → Add environment variables from your .env file
#    → Deploy

# 3. For Streamlit UI: go to share.streamlit.io → Deploy from GitHub (free)
```

---

## 11. Agent Build Instructions

If you are an AI coding agent using this README, follow these rules exactly:

1. **Create the full folder structure first** — run the `mkdir` command from Section 2 before writing any Python.
2. **Create all `__init__.py` files** — every folder under `ingestion/`, `rag/`, `llm/`, `api/`, `visualisation/`, `frontend/` needs one.
3. **Run `scripts/setup_db.py` before any ingestion** — fix all database connection errors before proceeding.
4. **Download at least 3 real `.nc` files** from `https://data-argo.ifremer.fr` into `data/raw/` before testing the parser.
5. **Complete each phase's deliverable check before starting the next phase** — do not write Phase 2 code until Phase 1 inserts rows successfully.
6. **Never hardcode API keys** — always use `os.getenv()` and load from `.env` with `python-dotenv`.
7. **Re-run `scripts/build_index.py` every time new data is ingested** — the FAISS index must reflect current data.
8. **Wrap every LLM-generated SQL in `safe_sql_query()`** before executing against the database.
9. **Run `pytest tests/ -v` and fix all failures before deployment.**
10. **Never commit `.env`** — confirm it is listed in `.gitignore` before the first `git push`.

### Quick-start command sequence (run in order)
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/setup_db.py

# Data
python scripts/run_ingestion.py
python scripts/build_index.py

# Run
uvicorn api.main:app --reload &
streamlit run frontend/app.py
```

If all commands succeed and `http://localhost:8501` shows a chat interface, the build is complete.

---

*SeaBorg — Making ocean data accessible to everyone.*