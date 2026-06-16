# SeaBorg: Current Project Status

## High-Level Architecture Overview
SeaBorg is designed to allow users to query ARGO oceanographic float data using plain English. The architecture is split into an offline ETL pipeline and a runtime RAG (Retrieval-Augmented Generation) pipeline exposed via a modern REST API. 

The project has successfully completed Phases 0 through 3. Below is the mapping of our current data flow and architecture.

### 1. Data Ingestion & ETL (Phase 0 & 1)
The offline pipeline is fully operational:
* **`ingestion/downloader.py`**: Fetches raw NetCDF (`.nc`) files from the ARGO FTP servers.
* **`ingestion/parser.py` & `qc_filter.py`**: Parses the raw files using `xarray` and `pandas`, applies necessary quality control flags (QC=1), and filters out anomalous range values for temperature, salinity, and depth.
* **`ingestion/db_loader.py`**: Loads the cleaned data into the `seaborg` PostgreSQL database (`argo_profiles` table) and saves a master Parquet file (`data/processed/argo.parquet`) which serves as the source of truth for the RAG pipeline.

### 2. RAG Pipeline & Vector Search (Phase 2 Achievements)
Phase 2 successfully established our semantic search capabilities. While previously undocumented, the `rag/` module is fully implemented:
* **Summarisation (`summariser.py`)**: Converts raw Parquet data rows into English sentences (e.g., "Float [id] recorded a temperature of...").
* **Embedding (`embedder.py`)**: Uses the local `all-MiniLM-L6-v2` sentence-transformer model to convert summarized text into 384-dimensional dense vectors.
* **Indexing (`indexer.py`)**: Builds a local `FAISS IndexFlatL2` vector index from the embeddings.
* **Retrieval (`retriever.py`)**: Maintains a module-level state (`_index` and `_df`). It embeds incoming user queries, performs a FAISS similarity search, and retrieves the top-K matching records directly from the Parquet DataFrame.

### 3. API Routes & Flow
The backend is powered by FastAPI (`api/main.py`). The flow for an incoming user query operates as follows:
* **Startup**: FastAPI triggers an `@app.on_event("startup")` hook to verify the PostgreSQL connection and load the FAISS index + Parquet DataFrame into memory via `retriever.load_index()`.
* **Routing**: The application exposes routers under the `/api` prefix for `/chat`, `/data`, and `/export`.
* **Execution**: Incoming chat queries route to the LLM pipeline, where the embedded user query retrieves context from FAISS, filters the data, and passes it to the `query_engine.py` to construct a fully grounded AI response and a matching SQL query.

### 4. Hybrid Retrieval Architecture (Phase 3A)
Introduced a new Query Router and Structured Query Engine to bypass FAISS for deterministically structured queries involving precise depths and geographic bounds.

### 5. Runtime Routing Integration (Phase 3A)
The API endpoint `/api/chat` now features active routing capabilities, smoothly serving both strictly deterministic tabular data and semantic LLM-grounded insights based on query classification.
