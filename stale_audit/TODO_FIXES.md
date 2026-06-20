# TODO Fixes

[ ] **Critical fixes**
- [ ] Migrate `structured_query/engine.py` from Pandas to PostgreSQL to prevent OOM on large filters.
- [ ] Remove `async` from `def chat()` in `api/routes/chat.py` to unblock the FastAPI event loop.
- [ ] Rewrite `/api/export` to use chunked streaming directly from the database cursor.
- [ ] Stop loading the Parquet file twice into memory (currently in both `engine.py` and `retriever.py`).

[ ] **High-priority fixes**
- [ ] Replace FAISS RAG for numerical data with a Text-to-SQL or Text-to-Pandas agent.
- [ ] Replace Regex/Keyword routing in `query_router.py` with an LLM classifier.
- [ ] Add pagination (`limit` / `offset`) to `/api/float/{float_id}`.
- [ ] Mock Groq API calls and FAISS loading in `tests/test_api.py` so CI passes offline.

[ ] **Medium-priority fixes**
- [ ] Remove the redundant `generate_sql()` call from the semantic RAG path in `query_engine.py`.
- [ ] Cache the results of `/api/stats` to avoid slow `COUNT(DISTINCT)` database hits.
- [ ] Merge `detect_chart_type` and `detect_visualization_intent` in `chat.py`.

[ ] **Low-priority fixes**
- [ ] Replace `print()` with Python's `logging` module throughout the codebase.
- [ ] Remove the unused `safe_sql_query` function from `nl_to_sql.py`.
- [ ] Use `pathlib.Path.resolve()` for `.env` file paths to prevent traversal.
- [ ] Rewrite `README.md` to document the current API endpoints instead of the build tutorial.
