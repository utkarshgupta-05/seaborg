# Improvement Roadmap

## Phase 1: Stability & Security (Immediate)
1. **Fix Synchronous Blocking:** Remove `async` from `api/routes/chat.py` or use `asyncio.to_thread` for all Pandas and FAISS operations so the API can handle multiple users.
2. **Implement Pagination:** Add `LIMIT` and `OFFSET` to `/api/float/{float_id}` to prevent JSON parsing OOMs.
3. **Remove Dead SQL Exec Code:** Delete the unused and unsafe `safe_sql_query` function from `nl_to_sql.py`.
4. **Fix Export Streaming:** Rewrite `/api/export` to use server-side database cursors and chunked string yields, instead of buffering the entire dataset in RAM.

## Phase 2: Architecture Unification (Short-term)
5. **Eliminate the Split Brain:** Migrate all Structured Query filtering logic from Pandas (`engine.py`) to PostgreSQL (`data.py` style). 
6. **Consolidate RAM Usage:** Stop loading the Parquet file twice on startup. Have a single data-loading module if Pandas must be used, or deprecate Pandas entirely in favor of SQL.
7. **Cache Heavy Queries:** Implement caching for `/api/stats` to prevent `COUNT(DISTINCT)` from choking the database on every page load.

## Phase 3: AI & RAG Quality (Medium-term)
8. **Replace Keyword Routing:** Implement an LLM intent classifier (or fine-tuned small model) in `query_router.py` to prevent ambiguous queries from being misrouted.
9. **Fix Tabular RAG:** Transition numerical and geo-bounding search away from FAISS L2 similarity. Use an Agentic framework where the LLM writes and executes SQL, then summarizes the resulting rows, rather than trying to embed numbers.
10. **Increase Context Windows:** If using LLM summarization, aggregate the data in SQL first (e.g., averages, groupings) and pass the *aggregations* to the LLM, rather than sending a random sample of `top_k=5` raw floats.

## Phase 4: Engineering Standards (Long-term)
11. **Refactor the "God Route":** Break `api/routes/chat.py` into smaller service functions (e.g., `services/visualization.py`, `services/router.py`).
12. **Mocked Unit Tests:** Refactor `tests/` to use `unittest.mock.patch` for the Groq client and FAISS index so they run instantly in CI/CD without side effects.
13. **Observability:** Replace all `print()` statements with structured JSON logging (`logger.info`) compatible with Datadog or CloudWatch.
