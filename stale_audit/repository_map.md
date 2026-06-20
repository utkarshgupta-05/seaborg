# Repository Map

## Overview
SeaBorg is structured as a full-stack Python (FastAPI) and React application, focusing on Ocean Data Analysis (ARGO floats) with both structured database querying and semantic Retrieval-Augmented Generation (RAG).

## Top-Level Directories

### `api/`
- **Purpose:** FastAPI backend application entry point and route definitions.
- **Key Files:** 
  - `main.py`: FastAPI app initialization, middleware (CORS), and router inclusion.
  - `routes/chat.py`: The core endpoint (`/api/chat`) that handles user messages and invokes the router, structured engine, and semantic engine.
- **Connections:** Receives requests from `frontend/`, calls `router/`, `structured_query/`, and `rag/`.

### `frontend/`
- **Purpose:** React application for the user interface. Built with Vite and TypeScript.
- **Key Files:** 
  - `src/`: React components, hooks, and API client.
  - `vite.config.ts`, `package.json`: Build and dependency configuration.
- **Connections:** Communicates via HTTP requests to `api/main.py` on the backend.

### `router/`
- **Purpose:** Classifies user natural language queries.
- **Key Files:**
  - `query_router.py`: Contains regex and keyword matching logic to determine if a query should be handled as `STRUCTURED` or `SEMANTIC`.

### `rag/`
- **Purpose:** The Semantic / Retrieval-Augmented Generation pipeline.
- **Key Files:**
  - `embedder.py`: Handles loading the SentenceTransformer model (`all-MiniLM-L6-v2`) and generating embeddings.
  - `indexer.py`: Logic for building FAISS indices.
  - `retriever.py`: Searches the FAISS index to return top-k matches from the dataset.
  - `summariser.py`: Formats retrieved data into LLM prompts to generate final answers.

### `structured_query/`
- **Purpose:** Handles analytical queries (filtering by depth, geo-bounding boxes, aggregating statistics).
- **Key Files:**
  - `engine.py`: Uses Pandas to filter the raw Parquet datasets based on structured query outputs.

### `visualisation/`
- **Purpose:** Generates visual output artifacts (charts, maps) for structured queries.
- **Key Files:**
  - `profile_chart.py`: Depth/Temperature profiles.
  - `timeseries_chart.py`: Aggregated trends over time.
  - `map_chart.py`: Geospatial distribution of floats.
  - `exporter.py`: Generates HTML/JSON formats for the frontend.

### `ingestion/`
- **Purpose:** ETL pipeline for loading, cleaning, and formatting ARGO float data.
- **Key Files:**
  - `parser.py`: Parses raw dataset formats (e.g. NetCDF/CSV).
  - `qc_filter.py`: Quality control filtering.

### `llm/`
- **Purpose:** Auxiliary LLM integration scripts (currently separated from core `rag` components).
- **Key Files:** `nl_to_sql.py`, `geo_mapping.py`, `prompts.py`, `query_engine.py`.

### `data/` & `indexes/`
- **Purpose:** Storage for local data files.
- **Key Files:** 
  - Parquet dataset files (e.g., `argo_profiles.parquet`).
  - FAISS index files (e.g., `argo_faiss.index`).
- **Concerns:** Large files stored locally may not scale well if deployed to ephemeral serverless environments like Render without persistent disks.

### `tests/`
- **Purpose:** Pytest suite containing unit and integration tests.
- **Key Files:** `test_api.py`, `test_rag.py`, `test_structured_query.py`, `test_router_integration.py`.

### `scripts/`
- **Purpose:** Helper scripts for developers.
- **Key Files:** `build_index.py`, `data_report.py`, `run_ingestion.py`.

### `notebooks/`
- **Purpose:** Jupyter notebooks for exploration and prototyping.

### Root Files
- `README.md`, `SEABORG_AGENT_README.md`: Primary documentation.
- `.env`, `.env.example`: Environment variable configurations.
- `requirements.txt`: Python dependencies.

## Key Concerns & Risks
1. **Memory constraints:** Relying on Pandas dataframes loaded directly into memory (`retriever.py`, `engine.py`) may cause Out of Memory (OOM) exceptions at scale.
2. **Coupling in API layer:** `api/routes/chat.py` might hold too much orchestration logic instead of delegating to a dedicated service layer.
3. **Router Naivety:** `query_router.py` relies heavily on keyword matching and regex, which is brittle and easily bypassed by ambiguous or complex user phrasing.
