# Code Quality Review

## Overview
The codebase exhibits signs of rapid prototyping and agent-driven development. While functional for small datasets, it carries significant technical debt.

## Code Smells & Issues

### 1. Dead Code (High)
**Location:** `llm/nl_to_sql.py`
**Issue:** The function `safe_sql_query` is defined and explicitly documented as a security guardrail, but it is **never called anywhere in the executable application**. The `chat.py` endpoint simply displays the generated SQL as metadata. Leaving dead, untested security functions in the codebase is misleading to other developers who might assume the system is executing SQL safely.

### 2. Duplicate Logic (Medium)
**Location:** `api/routes/chat.py`
**Issue:** `detect_chart_type(message)` and `detect_visualization_intent(message)` do almost exactly the same thing. They contain duplicate lists of keywords (`map_keywords`, `profile_keywords`, `timeseries_keywords`). If a developer wants to add a new chart type (e.g., "histogram"), they have to update it in multiple places.

### 3. Overloaded API Handlers (Medium)
**Location:** `api/routes/chat.py`
**Issue:** The `/api/chat` route is a "God function". It is over 100 lines long and handles routing logic, dataframe manipulation, visualization extraction, JSON sanitization (`sanitize_plotly_json`), and orchestrating multiple sub-engines. 
**Fix:** Move visualization processing to `visualisation/engine.py` and JSON sanitization to a utility file. The route should simply call a `handle_chat_request()` service function.

### 4. Poor Use of Globals (High)
**Location:** `rag/retriever.py` and `structured_query/engine.py`
**Issue:** Both files rely on `global _df` and `global _index` variables. Global state makes unit testing nearly impossible (state leaks between tests) and prevents the application from gracefully reloading data or serving multiple datasets.

### 5. Print Statements Instead of Logging (Low)
**Location:** `router/query_router.py`, `api/routes/chat.py`
**Issue:** The code uses `print("[ROUTER] STRUCTURED")` instead of Python's standard `logging` library. This means logs cannot be easily filtered by severity, exported to observability tools, or turned off in production.

## Recommendations
1. **Delete Dead Code:** Remove `safe_sql_query` or implement it properly.
2. **Refactor `chat.py`:** Extract the duplicate chart detection logic and visualization payload generation into separate service modules.
3. **Remove Globals:** Use dependency injection for the FAISS index and Parquet dataframe (e.g., FastAPI's `Depends()`).
4. **Implement standard logging:** Replace all `print()` calls with `logger.info()`.
