# Phase 3 Explanation: LLM Integration & Data Grounding

## Core Objectives
Phase 3 integrates the LLM (currently powered by Groq) with the RAG pipeline to produce safe, strictly grounded text answers and SQL queries. The primary goal of this phase is to ensure the AI behaves like a strict data analyst—summarizing retrieved context without hallucinating, inferring missing data, or making geographical assumptions.

## Key Components & New Functionalities

### 1. Query Engine (`llm/query_engine.py`)
This is the main orchestrator for LLM inference.
* **Data-Grounded Summarization**: The engine passes the user's question and retrieved DataFrame rows to the Groq LLM (defaulting to `llama-3.1-8b-instant`). 
* **Determinism**: The temperature is explicitly set to `0.1` to ensure analytical precision over creativity.
* **Null-Case Fallback Handling**: A critical safety guard is placed before the LLM is ever called. If the retrieved `context_rows` DataFrame is empty or `None`, the engine short-circuits and immediately returns a deterministic message: `"No data found for the requested region."`. This prevents the LLM from trying to invent an answer for empty queries.

### 2. Strict Prompting (`llm/prompts.py`)
The system prompts have been heavily constrained.
* **No Geographic Inference**: The LLM is explicitly instructed *not* to deduce whether a coordinate belongs to a specific ocean. The raw coordinates (lat/lon) have been removed from the context rows to prevent the LLM from reverse-engineering locations.
* **Row Limiting**: To prevent context window bloat and maintain LLM focus, `build_prompt` limits the context injected into the prompt to a maximum of 10 rows.

### 3. Geographic Mapping (`llm/geo_mapping.py`)
A brand new module introduced to handle region-based queries logically rather than probabilistically.
* **Bounding Boxes**: Maps human-readable strings (e.g., "Indian Ocean", "Bay of Bengal") to strict latitude and longitude boundary dictionaries.
* **Detection Engine**: The `detect_region` function scans user questions using a longest-match-first strategy to correctly identify requested regions before the text is parsed into SQL.

### 4. Natural Language to SQL (`llm/nl_to_sql.py`)
Translates plain English queries into valid PostgreSQL queries against the `argo_profiles` schema.
* **Pre-processing / Hint Injection**: Before generating SQL, `_preprocess_question` checks for named regions using `geo_mapping.py`. If a region is detected, it silently appends a coordinate hint to the LLM (e.g., `latitude BETWEEN 5 AND 25 AND longitude BETWEEN 80 AND 100`). This shifts the burden of geographic filtering to standard SQL rather than relying on the LLM's geographical knowledge.
* **Safe SQL Execution Guard**: The generated SQL is intercepted and checked against `FORBIDDEN_KEYWORDS` (e.g., `DROP`, `DELETE`, `UPDATE`). If a keyword is detected, it is firmly rejected with an `"Unsafe SQL rejected"` tuple to prevent malicious injections or destructive behavior.

## Verification Pipeline (`tests/verify_phase3.py`)
The phase is validated using `verify_phase3.py`, which mimics the exact flow of the API endpoint. It initiates the FAISS index, submits a semantic question (e.g., *"what is the average temperature at 200m depth?"*), retrieves relevant context, and invokes the `answer_query` function. The test ensures that both a plain English response (grounded exclusively in the retrieved metrics) and a safe `SELECT` statement are generated successfully.

## Phase 3A: Query Router + Structured Query Engine
To resolve the semantic limitations of FAISS on structured geographical and numerical queries (like exact depths), we introduced a **Query Router**.

* **`router/query_router.py`**: Intercepts user queries and classifies them as either `STRUCTURED` or `SEMANTIC` based on keyword detection.
* **`structured_query/engine.py`**: Bypasses FAISS entirely and loads the raw `argo.parquet` DataFrame. It extracts numerical and geographic constraints (using `geo_mapping.py`), filters the DataFrame deterministicly, and returns a human-readable statistical summary alongside the raw DataFrame and constraint metadata.

## Phase 3A: Runtime Integration
Successfully integrated the Query Router directly into the `POST /api/chat` endpoint. The API now intelligently checks `classify_query()`. If a query is `STRUCTURED`, it calls `answer_structured_query()` and bypasses FAISS completely, ensuring pure deterministic answers for depths and geographic filters. Semantic queries natively fall back to the existing FAISS + LLM pipeline.
