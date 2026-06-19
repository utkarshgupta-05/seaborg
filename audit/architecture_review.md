# Architecture Review

## System Architecture Overview
The SeaBorg application is built as a monolithic FastAPI backend serving a React frontend. The backend accepts HTTP requests, routes natural language questions, processes data, and returns JSON payloads containing text answers and Plotly visualization specifications.

### Frontend-to-Backend Flow
1. User enters a query in the React frontend.
2. The query is POSTed to the `/api/chat` endpoint.
3. The frontend receives JSON containing an answer, visualization data (JSONified Plotly figures), and metadata.
4. The frontend renders the chart using Plotly.js and displays the textual answer.

### Backend Internal Flow (The "Split Brain" Problem)
The backend architecture is highly fragmented, suffering from a **Split Brain Data Layer**.
- **Chat/QA Flow (`/api/chat`):** Relies on an in-memory Pandas dataframe loaded from a local Parquet file (`rag.retriever` and `structured_query.engine`).
- **Data APIs (`/api/floats`, `/api/export`):** Relies on a PostgreSQL relational database (`DATABASE_URL`). 

This means data must be synchronized between two entirely different storage paradigms, leading to inevitable inconsistencies, duplicated ETL efforts, and diverging schemas.

### Query Resolution Paths (Chat)
When a message arrives at `/api/chat`, it passes through `classify_query()` (in `router/query_router.py`).
1. **Structured Query Path:** If classified as STRUCTURED, the query goes to `structured_query.engine`. This engine uses an LLM to generate pandas filtering code or SQL, applies it to the in-memory dataframe, and returns results.
2. **Semantic (RAG) Path:** If classified as SEMANTIC, the query goes to `rag.retriever`. The query is embedded (via Hugging Face or a local model), searched against a local FAISS index, and the nearest neighbors are injected into an LLM prompt (`rag.summariser`) to generate a response.

---

## Architectural Weaknesses & Risks

### 1. Split-Brain Data Layer (Critical)
**Issue:** The application uses PostgreSQL for exports and listing floats, but a local Parquet file in Pandas for RAG and Structured querying.
**Impact:** Maintaining two sources of truth is dangerous. The local Parquet file is loaded entirely into RAM, which completely defeats the purpose of having a PostgreSQL database.

### 2. Tight Coupling in the API Layer (High)
**Issue:** The `/api/chat` endpoint directly handles visualization logic, orchestrates query classification, executes retrievals, and formats responses. 
**Impact:** `api/routes/chat.py` acts as a "God object" rather than just an HTTP transport layer. Visualization generation (`generate_visualization_payload`) has no business being tightly coupled directly inside the HTTP route.

### 3. Missing Abstraction Boundaries (Medium)
**Issue:** Data access is hardcoded. `rag.retriever` uses global `_df` and `_index` variables. `data.py` hardcodes raw SQL queries against PostgreSQL.
**Impact:** It is nearly impossible to write unit tests without spinning up a real database or loading the entire Parquet file. There is no dependency injection or Repository pattern.

### 4. Duplicate Intent Logic (Low)
**Issue:** `chat.py` contains `detect_chart_type()` and `detect_visualization_intent()`, which both do exactly the same regex/keyword matching to identify if a query wants a map, profile, or timeseries.
**Impact:** Code duplication and maintenance overhead. If a new chart type is added, it must be updated in multiple disjointed functions.
