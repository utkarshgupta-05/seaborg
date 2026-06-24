# ⚓ SeaBorg — Ocean Data Query Platform

SeaBorg is a hybrid ocean-data assistant for querying ARGO float observations through a combination of **structured database queries**, **semantic retrieval**, and **LLM-assisted summarization**.

It is built to answer questions about ocean profiles, depth ranges, temperature, salinity, oxygen, dates, and geographic regions with a code-first architecture that keeps exact numerical answers grounded in the data.

---

## ✨ Features

### 🔎 Intelligent Query Routing
- Routes each question into one of three paths:
  - **Structured** — direct PostgreSQL queries for exact filters and statistics
  - **Semantic** — FAISS-based retrieval for contextual matches
  - **Hybrid** — combines both paths for grounded answers with supporting context

### 📊 Structured Ocean Analytics
- Parses natural-language filters such as:
  - depth ranges
  - named ocean/sea regions
  - year-based date constraints
  - requested variables like temperature, salinity, and oxygen
- Queries the `argo_profiles` table through SQLAlchemy
- Returns deterministic summaries and matching rows

### 🧠 Semantic Retrieval
- Uses a local `sentence-transformers` embedding model
- Builds and loads a FAISS index over processed ARGO records
- Retrieves relevant profiles from the parquet corpus for contextual answers

### 🤝 Hybrid Answering
- Produces authoritative statistics from PostgreSQL
- Adds supporting semantic records from FAISS retrieval
- Sends both to Groq-powered LLM prompts for concise grounded responses

### 🗂️ Data Access APIs
- `/chat` for the main question-answer flow
- `/data` endpoints for direct exploration of ARGO records
- `/export` for downloading filtered results as CSV or NetCDF

### 🖥️ React Frontend
- Vite + React + TypeScript UI
- Search bar, answer panel, query history, sidebar, and chart panel
- Plotly-based visualizations for returned results

---

## 🏗️ Architecture

```text
seaborg-main/
├── api/
│   ├── main.py              # FastAPI app entrypoint
│   ├── database.py          # SQLAlchemy engine/config
│   ├── models.py            # Request/response schemas
│   └── routes/
│       ├── chat.py          # Main query endpoint
│       ├── data.py          # Data exploration endpoints
│       └── export.py        # CSV / NetCDF export endpoint
│
├── structured_query/
│   ├── parser.py            # NL filter parsing
│   ├── repository.py        # PostgreSQL access layer
│   ├── service.py           # Structured query orchestration
│   └── engine.py            # Public structured query entrypoint
│
├── rag/
│   ├── embedder.py          # SentenceTransformer embeddings
│   ├── indexer.py           # Builds FAISS index from parquet
│   ├── retriever.py         # FAISS retrieval + filtering
│   └── summariser.py        # Row-to-text summarization
│
├── retrieval/
│   └── hybrid_service.py    # Hybrid structured + semantic flow
│
├── llm/
│   ├── prompts.py           # Prompt templates
│   ├── query_engine.py      # RAG + LLM answer generation
│   ├── nl_to_sql.py         # SQL generation from questions
│   ├── context_builder.py   # Hybrid prompt construction
│   ├── geo_mapping.py       # Region → coordinate mapping
│   └── formatters.py        # Row formatting helpers
│
├── ingestion/
│   ├── downloader.py        # Data acquisition
│   ├── parser.py            # Raw ARGO parsing
│   ├── qc_filter.py         # Quality-control filtering
│   └── db_loader.py         # Load processed data into PostgreSQL
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── types/
│   └── vite.config.ts
│
├── data/
│   └── processed/argo.parquet
├── indexes/
│   └── argo.faiss
└── notebooks/
    ├── 01_explore_netcdf.ipynb
    ├── 02_test_rag.ipynb
    └── 03_visualisation_demo.ipynb
```

---

## 🔁 Query Flow

1. The frontend sends a question to the FastAPI backend through `/chat`.
2. `api/routes/chat.py` classifies the query using the router.
3. The router decides whether the request is:
   - **structured**
   - **semantic**
   - **hybrid**
4. Structured queries are parsed for filters and answered from PostgreSQL.
5. Semantic queries are embedded and matched against the FAISS index.
6. Hybrid queries combine structured statistics with semantic context.
7. The backend returns:
   - a natural-language answer
   - optional chart metadata
   - SQL used for display
   - confidence and diagnostic metadata

---

## 🧩 Key Technical Decisions

- **FastAPI** for a clean Python API layer
- **PostgreSQL** as the authoritative structured data source
- **FAISS** for fast semantic retrieval
- **SentenceTransformers** for local embeddings
- **Groq** for LLM-backed response synthesis
- **React + Vite + TypeScript** for the frontend
- **Parquet** as the retrieval corpus backing store

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL
- A Groq API key

### Backend Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```env
   DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/seaborg
   GROQ_API_KEY=your_groq_api_key
   LLM_MODEL=llama-3.1-8b-instant
   PARQUET_PATH=data/processed/argo.parquet
   FAISS_INDEX_PATH=indexes/argo.faiss
   FRONTEND_URL=http://localhost:5173
   ```

3. Make sure your PostgreSQL database contains the `argo_profiles` table and the processed ARGO data.

4. Start the backend:
   ```bash
   uvicorn api.main:app --reload
   ```

### Frontend Setup

1. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open the app in your browser at the Vite URL shown in the terminal.

---

## 🗄️ Data Pipeline

SeaBorg works with ARGO profile data that is processed into two main forms:

- **PostgreSQL** for exact querying and aggregation
- **Parquet + FAISS** for semantic retrieval

The ingestion layer handles raw data download, parsing, quality-control filtering, and loading into the database. The indexer builds the FAISS index from the processed parquet corpus so semantic queries can retrieve nearby profiles efficiently.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, SQLAlchemy, Python |
| Database | PostgreSQL |
| Retrieval | FAISS |
| Embeddings | SentenceTransformers |
| LLM | Groq |
| Data | Pandas, NumPy, Parquet, NetCDF tooling |
| Frontend | React, TypeScript, Vite |
| Visualization | Plotly |
| Testing | pytest, httpx |

---

## 📄 Project Structure Summary

- **`api/`** — backend app and HTTP endpoints
- **`structured_query/`** — exact query parsing and SQL execution
- **`rag/`** — semantic retrieval pipeline
- **`retrieval/`** — hybrid orchestration
- **`llm/`** — prompt building, SQL generation, and answer synthesis
- **`ingestion/`** — ARGO data processing pipeline
- **`frontend/`** — user-facing React app
- **`data/processed/`** — stored parquet dataset
- **`indexes/`** — FAISS vector index
- **`notebooks/`** — exploratory and demo notebooks

---

## 📌 Notes

SeaBorg is designed around a clean split between exact structured querying and flexible semantic retrieval, with the hybrid path acting as the main user-facing experience.

---

## 📄 License

Add your preferred license here.
