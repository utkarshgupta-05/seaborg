# Production Readiness Review

## Verdict: NOT READY FOR PRODUCTION

The SeaBorg application is currently a functioning **Proof of Concept (PoC)**. While it successfully demonstrates the integration of LLMs, FAISS, and React to answer oceanographic questions, its underlying architecture cannot support concurrent users, large datasets, or reliable LLM routing.

## Key Roadblocks to Production

### 1. The Split Brain Architecture
A production system must have a single source of truth. Currently, data lives in PostgreSQL (for exports) and in a local Parquet file (for chat queries). This guarantees data drift and doubles infrastructure costs.

### 2. OOM (Out of Memory) Risk at Scale
The application loads the entire Parquet dataset into Pandas memory on startup (twice). Any dataset larger than a few hundred megabytes will crash a standard web server container. Furthermore, exporting data buffers the entire dataset in RAM rather than streaming it, guaranteeing an OOM crash for large exports.

### 3. Concurrency Blocking
The `/api/chat` route is defined as `async def`, but executes synchronous, CPU-bound Pandas and FAISS operations. This blocks the FastAPI event loop, meaning SeaBorg can only handle exactly one user request at a time. A production API must be able to handle concurrent requests.

### 4. Flawed RAG Pipeline
Using a semantic embedding model to search tabular, numerical data ensures that the system will return incorrect results for queries involving specific numbers, inequalities (e.g., "deeper than 500m"), or dates. The system is extremely prone to hallucination because it feeds the LLM textually similar (but mathematically wrong) data rows.

### 5. Prompt Injection Susceptibility
Because the user's input is passed directly to the Groq LLM without sanitization, malicious users can hijack the LLM to output undesirable text or leak the system prompt.

## Path to Production
Before this app can be deployed to real users:
1. Move ALL data querying (structured and vector) to a single database (e.g., PostgreSQL with `pgvector`).
2. Implement true asynchronous routes or use background workers (Celery/RQ) for long-running LLM and data tasks.
3. Replace the Regex/Keyword Query Router with an LLM-based intent classifier.
4. Replace the RAG pipeline with a Text-to-SQL or Text-to-Pandas Agentic workflow for querying numerical data.
