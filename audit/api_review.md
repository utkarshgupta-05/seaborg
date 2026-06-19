# API Quality Review

## Overview
The backend exposes a FastAPI application with three primary route files: `chat.py`, `data.py`, and `export.py`.

## Endpoint Analysis

### `POST /api/chat` (in `api/routes/chat.py`)
- **Purpose:** Processes natural language queries and returns textual answers with visualizations.
- **Inputs:** `ChatRequest` (contains `message`).
- **Outputs:** `ChatResponse` JSON.
- **Failures & Risks:**
  - **No Global Exception Handling:** If `answer_structured_query` or `retrieve` throws an exception, it crashes the endpoint and returns a raw 500 error to the frontend.
  - **Blocking Operations:** It performs heavy Pandas processing and FAISS similarity searches synchronously inside an `async def` function. This blocks the FastAPI event loop, freezing the entire server for all other users during query execution.
  - **Overloaded Responsibilities:** Evaluates visualization intent, formats Plotly JSON, and orchestrates the LLM all within the route handler.

### `GET /api/floats` (in `api/routes/data.py`)
- **Purpose:** Paginated list of unique float IDs with bounding box and date ranges.
- **Inputs:** `page` (default 1), `page_size` (default 20).
- **Outputs:** JSON list of floats and total count.
- **Failures & Risks:**
  - **SQL Injection Risk:** Safely uses parameterized queries (`:limit`, `:offset`), so injection risk is low here.
  - **Performance Risk:** `COUNT(DISTINCT float_id)` over the entire `argo_profiles` table is extremely slow on large datasets. 

### `GET /api/float/{float_id}` (in `api/routes/data.py`)
- **Purpose:** Retrieves all readings for a specific ARGO float, with optional depth and date filters.
- **Inputs:** `float_id` (path param), `start_date`, `end_date`, `depth_min`, `depth_max` (query params).
- **Outputs:** JSON list of rows.
- **Failures & Risks:**
  - **Unbounded Response Size:** If a float has 100,000 readings, this endpoint attempts to fetch and return all 100,000 rows at once in JSON format. It will crash the server (OOM) or timeout. Pagination is desperately needed.
  - **No Input Validation on Dates:** `start_date` and `end_date` accept raw strings without validating ISO 8601 format, leading to potential PostgreSQL casting errors.

### `GET /api/stats` (in `api/routes/data.py`)
- **Purpose:** Aggregate statistics for the entire database.
- **Outputs:** JSON dictionary of min/max dates, bounding boxes, and counts.
- **Failures & Risks:**
  - **Database Bottleneck:** Running full table aggregations (`MIN`, `MAX`, `COUNT`) dynamically on every API call is a severe performance bottleneck. These statistics should be cached or materialized in PostgreSQL.

### `POST /api/export` (in `api/routes/export.py`)
- **Purpose:** Streams a dataset download in CSV or NetCDF format.
- **Inputs:** `ExportRequest` (`float_ids`, `format`, `start_date`, `end_date`).
- **Outputs:** `StreamingResponse`.
- **Failures & Risks:**
  - **Fake Streaming:** The endpoint uses `StreamingResponse`, but it literally loads the *entire* dataset into memory (`pd.read_sql`), converts the *entire* dataset to a massive string buffer in RAM (`df.to_csv(buffer)`), and then yields the buffer all at once (`iter([buffer.getvalue()])`). This entirely defeats the purpose of streaming and will cause OOM crashes on large exports.
  - **SQL Injection via ANY:** `float_id = ANY(:float_ids)` is generally safe via SQLAlchemy params, but the endpoint does not limit how many `float_ids` can be requested at once.

## Recommended Improvements
1. **Remove blocking calls from `async def chat`.** Change it to `def chat` to run in a threadpool, or make the underlying functions truly asynchronous.
2. **Add pagination** to `GET /api/float/{float_id}`.
3. **Fix the fake streaming** in `POST /api/export` by streaming directly from the database cursor in chunks using `yield`.
4. **Implement a Global Exception Handler** to return standardized HTTP 400/500 JSON responses with user-friendly error messages instead of stack traces.
