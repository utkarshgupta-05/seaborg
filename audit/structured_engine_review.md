# Structured Engine Review

## Overview
The Structured Query Engine (`structured_query/engine.py`) is responsible for deterministic filtering of the ARGO floats dataset using Pandas. When a query is routed to this engine, it attempts to extract depth and geographical constraints and apply them to the dataset.

## Logic Analysis

### Depth Extraction (`extract_depth`)
Depth bounds are extracted using simple Regular Expressions instead of an NLP entity extractor.
- `between X and Ym` -> depth `X` to `Y`
- `below Xm` -> depth `X` to infinity
- `above Xm` -> depth 0 to `X`
- `at Xm` -> depth `X - 50` to `X + 50`

### Geographic Extraction (`llm/geo_mapping.py`)
Despite being located in the `llm/` directory, `geo_mapping.py` **does not use an LLM.** It is a hardcoded dictionary mapping exactly 7 regions (Indian Ocean, Pacific Ocean, Atlantic Ocean, Arabian Sea, Bay of Bengal, South China Sea, Mediterranean Sea) to static bounding boxes. It does a simple substring match (`if name in text_lower`).

## Weaknesses & Flaws

### 1. Extremely Fragile Parsing (High Risk)
Because depth extraction relies on precise regex matches like `r"between\s+(\d+)\s*(?:and|to|-)\s*(\d+)\s*m\b"`, it will fail if a user writes:
- *"between 100 meters and 200 meters"* (fails because it expects "m", not "meters")
- *"100m - 200m"* (fails because it expects "between")
- *"deeper than 500m"* (fails because it expects "below" or "greater than")
When extraction fails, the engine applies no filters and queries the entire global dataset.

### 2. Extremely Limited Geographic Support (High Risk)
The hardcoded 7 bounding boxes represent a tiny fraction of oceanographic queries. 
- If a user asks about the "Gulf of Mexico", "North Sea", or "Southern Ocean", the engine will not filter geographically at all. It will silently return global data, which the user will assume is specific to their requested region.
- Hardcoded bounding boxes for oceans are also incredibly coarse and do not account for complex coastlines or exact oceanographic boundaries.

### 3. Misleading Result Summaries (Medium Risk)
If the engine fails to parse the user's constraints, it silently ignores them and runs a global query. The output summary says *"Found X matching observations. Depth range: 0m-2000m. Temperature range..."*. To a non-technical user, this looks like a successful response, but the data is completely wrong for their specific question.

### 4. Poor Handling of Empty Results (Low Risk)
If a filter results in 0 rows, the engine returns `summary = "Found 0 matching observations."` with an empty dataframe. The frontend chart generation then fails gracefully, but it does not attempt to explain *why* there is no data or suggest a broader query.

### 5. Memory Exhaustion Risk (Critical)
`engine.py` calls `filtered_df = _df[mask].copy()`. 
If a user requests the entire Atlantic Ocean without depth filters, it will create a deep copy of millions of rows in memory. On a server with 512MB RAM, this will cause an immediate Out of Memory (OOM) crash.

## Recommendations
1. **Replace Regex with LLM Entity Extraction:** Use an LLM function call (e.g., via `instructor` or LangChain) to reliably extract `min_depth`, `max_depth`, `min_lat`, `max_lat`, `min_lon`, `max_lon` from the natural language query.
2. **Move away from Pandas:** Execute these structured queries against PostgreSQL (`api/routes/data.py` style) instead of doing in-memory Pandas filtering. This solves the memory exhaustion risk.
3. **Fail Loudly:** If the system detects a geo-keyword but cannot resolve the bounding box, it should return an error stating "Region not recognized" rather than silently returning global data.
