# Executive Summary
- **What the project does:** SeaBorg is an AI-powered conversational web application that allows users to query and visualize ARGO float oceanographic data (temperature, salinity, depth) using natural language. It combines a React frontend with a FastAPI backend, utilizing both structured database filtering and a semantic Retrieval-Augmented Generation (RAG) pipeline.
- **Overall health:** The project is a functional Proof of Concept (PoC) but is structurally fragile and not ready for production. It suffers from a "split-brain" data architecture, blocking concurrent requests, and fundamentally misapplies semantic vector search to numerical tabular data.
- **Main strengths:** The user interface concept is excellent, seamlessly blending chat with dynamic Plotly visualizations. The underlying data ingestion pipeline successfully normalizes complex NetCDF files into queryable formats.
- **Main risks:** Severe memory exhaustion (OOM) risks at scale due to loading massive dataframes into RAM. The semantic RAG pipeline is prone to hallucination because FAISS embeddings cannot accurately parse mathematical inequalities or geographic coordinates.

# Scores
- **Architecture:** 3/10
- **API Quality:** 4/10
- **Query Router:** 2/10
- **Structured Engine:** 4/10
- **RAG Pipeline:** 2/10
- **Security:** 5/10
- **Performance:** 2/10
- **Testing:** 3/10
- **Documentation:** 4/10
- **Production Readiness:** 1/10

# Critical Issues

### Split-Brain Data Layer
- **Severity:** CRITICAL
- **File path:** `api/routes/data.py` vs `rag/retriever.py`
- **Function/class name:** System-wide
- **Description:** The app uses PostgreSQL for data exports but a local Parquet file loaded into Pandas for the chat endpoint.
- **Why it matters:** Guarantees data drift, doubles infrastructure costs, and prevents vertical scaling.
- **Recommended fix:** Centralize all querying directly onto PostgreSQL.

### Semantic Search on Numerical Data
- **Severity:** CRITICAL
- **File path:** `rag/embedder.py`, `rag/indexer.py`
- **Function/class name:** `embed_texts`
- **Description:** Attempting to use Hugging Face embeddings to search for numeric ranges (e.g., "deeper than 500m").
- **Why it matters:** Embedding models do not understand math. FAISS will retrieve rows based on textual similarity, returning mathematically incorrect data to the LLM, causing guaranteed hallucinations.
- **Recommended fix:** Replace FAISS RAG with a Text-to-SQL or Text-to-Pandas agent for tabular data.

### Synchronous Blocking of Event Loop
- **Severity:** CRITICAL
- **File path:** `api/routes/chat.py`
- **Function/class name:** `async def chat`
- **Description:** CPU-bound Pandas operations and synchronous Groq HTTP requests are executed directly inside an asynchronous FastAPI route.
- **Why it matters:** Freezes the entire server. SeaBorg can only handle exactly one concurrent user request at a time.
- **Recommended fix:** Change the route to a synchronous `def chat` so FastAPI offloads it to a threadpool, or make all underlying calls natively `async`.

### Memory Exhaustion (OOM) via Deep Copies
- **Severity:** CRITICAL
- **File path:** `structured_query/engine.py`
- **Function/class name:** `answer_structured_query`
- **Description:** `filtered_df = _df[mask].copy()` deep copies the filtered dataset in memory for every structured query.
- **Why it matters:** On a dataset with 1M+ rows, a single user requesting a large region will instantly crash the server with an Out of Memory error.
- **Recommended fix:** Execute structured filtering directly in PostgreSQL via SQL.

### Fake Export Streaming
- **Severity:** CRITICAL
- **File path:** `api/routes/export.py`
- **Function/class name:** `export_data`
- **Description:** Promises a `StreamingResponse`, but physically buffers the entire dataset into a massive RAM object (`buffer.getvalue()`) before yielding.
- **Why it matters:** Will instantly OOM crash the server on large exports.
- **Recommended fix:** Stream directly from a psycopg2 server-side cursor in chunks.

# High Priority Issues

### Keyword Router False Positives
- **Severity:** HIGH
- **File path:** `router/query_router.py`
- **Function/class name:** `classify_query`
- **Description:** Hardcoded keywords (e.g., "deeper", "trend") shunt semantic questions to the Structured engine, causing them to fail.
- **Why it matters:** Users cannot ask hybrid questions (e.g., "Explain the temperature trend at 500m").
- **Recommended fix:** Replace regex routing with an LLM intent classifier.

### Missing Pagination
- **Severity:** HIGH
- **File path:** `api/routes/data.py`
- **Function/class name:** `get_float`
- **Description:** Returns all historical rows for a float in one massive JSON payload.
- **Why it matters:** A float with 100,000 readings will crash the client's browser and the server's JSON encoder.
- **Recommended fix:** Implement `limit` and `offset` query parameters.

### Prompt Injection Risk
- **Severity:** HIGH
- **File path:** `llm/query_engine.py`
- **Function/class name:** `answer_query`
- **Description:** Raw user queries are injected into the LLM without sanitization.
- **Why it matters:** Malicious users can force the LLM to ignore instructions and leak system prompts or API keys.
- **Recommended fix:** Implement an input guardrail LLM step before querying.

### Unmocked Test Suite
- **Severity:** HIGH
- **File path:** `tests/test_api.py`
- **Function/class name:** Entire suite
- **Description:** Tests actually boot the FAISS index, connect to Postgres, and make live HTTP calls to Groq.
- **Why it matters:** The tests will fail in CI/CD environments without active databases or internet keys.
- **Recommended fix:** Use `unittest.mock.patch` to mock Groq responses and use SQLite for testing.

# Medium Priority Issues

### Redundant LLM Calls
- **Severity:** MEDIUM
- **File path:** `llm/query_engine.py`
- **Function/class name:** `answer_query`
- **Description:** `generate_sql()` is called on every semantic query, doubling latency and API costs, even though the SQL is never executed.
- **Why it matters:** Unnecessary latency and financial cost.
- **Recommended fix:** Remove SQL generation from the semantic path.

### Database Aggregations on Every Request
- **Severity:** MEDIUM
- **File path:** `api/routes/data.py`
- **Function/class name:** `get_stats`
- **Description:** Runs slow `COUNT(DISTINCT)` and `MIN/MAX` queries dynamically on every page load.
- **Why it matters:** Will choke the database CPU at scale.
- **Recommended fix:** Cache these stats in Redis or a materialized view.

### Duplicate Visualization Logic
- **Severity:** MEDIUM
- **File path:** `api/routes/chat.py`
- **Function/class name:** `detect_chart_type`, `detect_visualization_intent`
- **Description:** Identical copies of the same keyword logic.
- **Why it matters:** Maintenance overhead if new charts are added.
- **Recommended fix:** Merge into a single utility function.

# Low Priority Issues

### Unsafe Environment Variables
- **Severity:** LOW
- **File path:** `rag/indexer.py`
- **Function/class name:** `build_and_save`
- **Description:** Path traversal is possible if a malicious user controls the `.env` file.
- **Why it matters:** Minor risk since `.env` is server-side only.
- **Recommended fix:** Use `pathlib.Path.resolve()`.

### Print Statements instead of Logging
- **Severity:** LOW
- **File path:** `router/query_router.py`
- **Function/class name:** `classify_query`
- **Description:** Uses `print()` instead of `logging`.
- **Why it matters:** Prevents proper log filtering in production (Datadog/CloudWatch).
- **Recommended fix:** Use Python's `logging` module.

### Hardcoded Geo Regions
- **Severity:** LOW
- **File path:** `llm/geo_mapping.py`
- **Function/class name:** `detect_region`
- **Description:** Only supports 7 hardcoded regions. Returns global data if not recognized.
- **Why it matters:** Users will receive wrong data if they ask for a region outside the 7 hardcoded options.
- **Recommended fix:** Use an LLM to extract exact bounding boxes from the query.

# Top 10 Improvements
1. **Migrate all structured querying to PostgreSQL.**
2. **Convert `/api/chat` to a thread-safe, non-blocking route.**
3. **Replace FAISS RAG with an Agentic Text-to-SQL architecture.**
4. **Implement real chunked streaming for CSV/NetCDF exports.**
5. **Replace Regex Query Router with an LLM Intent Classifier.**
6. **Add pagination to the `/api/float` endpoint.**
7. **Mock the Groq LLM client and FAISS index in the Pytest suite.**
8. **Delete the unsafe, dead `safe_sql_query` function.**
9. **Remove redundant LLM SQL generation from the semantic RAG path.**
10. **Refactor the monolithic `/api/chat` function into smaller service classes.**

# Risk Assessment
- **What is most likely to break first:** The server will crash with an Out of Memory (OOM) error due to the Pandas deep copy in `engine.py` or the fake streaming in `export.py` as soon as the dataset grows.
- **What could cause incorrect answers:** The FAISS embedding search on numerical data will retrieve mathematically irrelevant rows for any specific depth/coordinate query, causing the LLM to hallucinate wrong answers confidently.
- **What could hurt scalability:** The single-threaded blocking of the FastAPI event loop means 1 concurrent user can freeze the entire application.
- **What could block deployment:** The tests failing in CI/CD due to lack of mocking.

# Portfolio / Resume Readiness
- **Can this be shown to recruiters?** Yes, as a Proof of Concept (PoC). The UI looks impressive, and integrating React with FastAPI and an LLM shows strong full-stack initiative.
- **What would make it stronger?** Fixing the "Split-Brain" architecture. A recruiter reviewing the code will instantly notice that Pandas and Postgres are duplicating the same data role.
- **What missing engineering signals are visible?** 
  - Lack of unit testing (only integration tests exist).
  - Lack of asynchronous concurrency awareness in FastAPI.
  - Lack of understanding of RAG limitations (trying to embed tabular numbers).

# Final Recommendation
**Major restructuring required.** The system is a visually appealing PoC, but the backend architecture is fundamentally unscalable and mathematically flawed for numerical RAG. The "Split-Brain" Pandas/Postgres duplication and the blocking event loops must be fixed before any real traffic can be served.
