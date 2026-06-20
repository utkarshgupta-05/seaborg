# Critical Issues

| Severity | Issue | Location | Description & Impact | Recommended Fix |
|----------|-------|----------|----------------------|-----------------|
| **CRITICAL** | Split-Brain Data Layer | `api/routes/data.py` vs `rag/retriever.py` | The app uses Postgres for exports but a local Parquet file for RAG/Chat. Causes data drift and doubles infrastructure costs. | Centralize all queries on PostgreSQL. If vector search is needed, use `pgvector`. |
| **CRITICAL** | Semantic Search on Numerical Data | `rag/embedder.py`, `indexer.py` | Embedding models (MiniLM) cannot understand numerical inequalities ("deeper than 1000m"). RAG returns mathematically incorrect rows based on textual similarity. | Replace FAISS RAG with a Text-to-SQL or Text-to-Pandas agent for tabular data. |
| **CRITICAL** | Synchronous Blocking of Event Loop | `api/routes/chat.py` (line 164) | `async def chat` executes CPU-bound Pandas operations and blocking Groq HTTP requests, freezing the server for all other concurrent users. | Change to `def chat` to run in a threadpool, or make all underlying DB/LLM calls asynchronous. |
| **CRITICAL** | Memory Exhaustion (OOM) | `structured_query/engine.py` (line 107) | `filtered_df = _df[mask].copy()` deep copies millions of rows in memory for every structured query. Will crash server at scale. | Execute structured filtering directly in PostgreSQL, not in memory via Pandas. |
| **CRITICAL** | Fake Export Streaming | `api/routes/export.py` (line 69) | Loads entire DB into Pandas, converts to a massive RAM buffer, and yields it at once. Will OOM on large exports. | Stream directly from a psycopg2 server-side cursor without buffering in Pandas. |

---

# High Priority Issues

| Severity | Issue | Location | Description & Impact | Recommended Fix |
|----------|-------|----------|----------------------|-----------------|
| **HIGH** | Keyword Router False Positives | `router/query_router.py` | Hardcoded keywords like "deeper" or "trend" immediately shunt semantic questions to the Structured engine, causing them to fail. | Replace regex routing with an LLM intent classifier or a hybrid routing approach. |
| **HIGH** | Missing Pagination | `api/routes/data.py` | `get_float()` returns all historical rows for a float in one JSON payload, potentially crashing the client and server. | Implement `limit` and `offset` query parameters. |
| **HIGH** | Prompt Injection Risk | `llm/query_engine.py` | Raw user queries are injected into the Groq LLM without sanitization. | Implement input sanitization or a system guardrail. |
| **HIGH** | Unmocked Test Suite | `tests/test_api.py` | Tests require a live database, FAISS index, and internet (Groq API) to pass. Fails in standard CI/CD. | Use `unittest.mock.patch` to mock Groq responses and use an in-memory SQLite DB for tests. |

---

# Medium Priority Issues

| Severity | Issue | Location | Description & Impact | Recommended Fix |
|----------|-------|----------|----------------------|-----------------|
| **MEDIUM** | Redundant LLM Calls | `llm/query_engine.py` (line 33) | `generate_sql()` is called on every semantic query even though the SQL is never executed, doubling latency and API costs. | Remove SQL generation from the semantic path, or only generate it if explicitly requested. |
| **MEDIUM** | Database Aggregations on every Request | `api/routes/data.py` (line 120) | `/api/stats` runs slow `COUNT(DISTINCT)` queries dynamically on every page load. | Cache these stats using Redis or materialize them in a view. |
| **MEDIUM** | Duplicate Visualization Logic | `api/routes/chat.py` | `detect_chart_type` and `detect_visualization_intent` are identical copies of the same keyword logic. | Merge into a single utility function. |

---

# Low Priority Issues

| Severity | Issue | Location | Description & Impact | Recommended Fix |
|----------|-------|----------|----------------------|-----------------|
| **LOW** | Unsafe Environment Variables | `rag/indexer.py` | Path traversal is possible if a malicious user controls `.env`. | Use `pathlib.Path.resolve()`. |
| **LOW** | Print Statements instead of Logging | `router/query_router.py` | `print()` prevents proper log filtering and observability in production. | Use Python's `logging` module. |
| **LOW** | Hardcoded Geo Regions | `llm/geo_mapping.py` | Only supports 7 regions. Returns global data if region is not recognized. | Fail explicitly if a region is not found, or use an LLM to extract true bounding boxes. |
