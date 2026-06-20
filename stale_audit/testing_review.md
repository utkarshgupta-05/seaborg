# Testing Review

## Overview
The `tests/` directory contains several pytest files (`test_api.py`, `test_structured_query.py`, etc.). The suite is small and functions primarily as a set of integration and end-to-end (E2E) tests rather than unit tests.

## Key Issues

### 1. No Mocking (High Risk)
The tests directly import `api.main.app` and use `TestClient(app)`. Because `main.py` has an `@app.on_event("startup")` hook that loads the FAISS index and connects to PostgreSQL, **running the test suite requires a fully populated local database, a built FAISS index, and a valid Parquet file.** 
Additionally, the tests make actual HTTP calls to the Groq API because the LLM clients are not mocked. If the `.env` file is missing `GROQ_API_KEY`, or if the internet is down, the tests will fail. This makes the tests completely unsuitable for CI/CD pipelines.

### 2. Missing Unit Tests for Critical Logic (High Risk)
- `router/query_router.py`: There are no isolated unit tests proving that specific ambiguous keywords route correctly.
- `rag/retriever.py`: No tests verify that the FAISS index returns the correct `top_k` matches or handles empty data gracefully.
- `structured_query/engine.py`: The `test_structured_query.py` file is actually just a script with `print()` statements! It doesn't use `pytest` assertions properly; it just runs a `for` loop and prints to stdout. This means CI cannot automatically detect failures in the structured query engine.

### 3. Fragile SQL Safety Test (Medium Risk)
`test_chat_endpoint_sql_safety` checks if "DROP TABLE" is returned by the LLM. Because it relies on the actual LLM's non-deterministic response, this test is "flaky". The LLM might randomly decide to output "I cannot drop tables" which passes the test, but doesn't prove the backend security logic works.

## Top 10 Places Where Tests Are Urgently Needed
1. **Mocked LLM tests:** Create a `unittest.mock.patch` for `Groq` so tests run offline.
2. **Query Router:** Unit tests for edge cases (e.g., "What is the temperature trend?").
3. **Regex Depth Extraction:** `engine.py` depth regex needs exhaustive unit testing (e.g., "100m", "100 meters", "below 50m").
4. **Geo Mapping:** Unit tests to ensure `detect_region` correctly handles overlapping strings (e.g., "Sea" vs "Arabian Sea").
5. **Pagination:** `data.py` `list_floats` needs tests to verify `page` and `page_size` bounds.
6. **Export Format:** `export_data` needs tests to verify the CSV and NetCDF byte streams are correctly formatted.
7. **Empty Data Handling:** Test what happens when the DB or Parquet file is empty.
8. **Invalid API Inputs:** Test `/api/chat` with empty strings, massive strings (10,000 chars), and malformed JSON.
9. **Visualisation Payloads:** Unit test `generate_visualization_payload` to ensure it safely handles NaNs or Infinities in the dataframe.
10. **SQL Validation:** If `safe_sql_query` is kept, it needs rigorous testing against clever SQL injections.
