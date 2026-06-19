# Performance Review

## Overview
The current architecture prioritizes rapid prototyping over performance. By relying entirely on loading large Parquet files into memory via Pandas and processing natural language synchronously on the main thread, the application faces severe performance bottlenecks under load.

## Bottlenecks

### 1. Synchronous Blocking on the Event Loop (Critical)
**Location:** `api/routes/chat.py` -> `async def chat()`
**Issue:** FastAPI uses an asynchronous event loop. By defining the endpoint as `async def` but calling blocking Pandas functions (`answer_structured_query`), FAISS CPU searches (`retrieve`), and synchronous HTTP requests (`Groq` client without `async`), the entire FastAPI server is frozen.
**Impact:** While one user's query is being processed (which could take 5-10 seconds for the LLM call), *no other user can access the API*. Concurrency is practically zero.

### 2. Double Memory Footprint for the Same Data (High)
**Location:** `rag/retriever.py` and `structured_query/engine.py`
**Issue:** Both files independently call `pd.read_parquet()` into separate module-level `_df` variables.
**Impact:** The dataset is loaded into RAM twice. This halves the available memory on the server.

### 3. Database Aggregations on Every Request (High)
**Location:** `api/routes/data.py` -> `get_stats()`
**Issue:** The `/api/stats` endpoint runs `MIN()`, `MAX()`, and `COUNT(DISTINCT)` across the entire PostgreSQL table on every single request.
**Impact:** `COUNT(DISTINCT)` is notoriously slow in PostgreSQL. On a large dataset, this endpoint will take seconds to return, consuming high CPU and DB IO.

### 4. Fake Streaming Exports (Medium)
**Location:** `api/routes/export.py` -> `export_data()`
**Issue:** The export endpoint promises a `StreamingResponse`, but physically loads the entire dataframe into memory, converts it to a massive `io.StringIO` buffer, and yields it all at once.
**Impact:** Exporting a large date range will crash the server immediately due to memory exhaustion.

## Recommendations
1. **Remove `async` from `chat()`:** Changing `async def chat` to `def chat` will force FastAPI to run the blocking code in a background threadpool, restoring concurrency. (Or better, rewrite the LLM and DB calls to use async libraries).
2. **Consolidate Data Loading:** Load the Parquet dataframe in a single shared module (or load it from Postgres) to avoid duplicating RAM usage.
3. **Cache the Stats:** Use Redis or in-memory caching for `/api/stats` since the underlying data updates infrequently.
