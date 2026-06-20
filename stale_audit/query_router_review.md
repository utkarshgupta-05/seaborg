# Query Router Review

## Overview
The query router (`router/query_router.py`) acts as the gatekeeper for `/api/chat`. It determines whether a natural language query requires exact database filtering/aggregations (`STRUCTURED`) or contextual answering based on semantic embeddings (`SEMANTIC`).

## Logic Analysis
The router relies entirely on hardcoded dictionary checks and regex matching against the user's query string, converted to lowercase.

**Structured Trigger Conditions:**
1. Aggregation keywords: `average, mean, min, max, count, total, highest, lowest`
2. Depth keywords: `above, below, deeper, shallow, depth`
3. Regex depth match: `r"\d+\s*m\b"` (e.g., "500m")
4. Geography keywords: `latitude, longitude, equator, north, south, east, west`
5. Visualization keywords: `profile, plot, chart, graph, map, location, trend, over time`

**Semantic Fallback:**
If *none* of the above keywords are found, the query is routed to the RAG Semantic path.

## Routing Weaknesses & Flaws

### 1. High False-Positive Rate for Structured Queries (High Risk)
Because the router simply checks `if any(k in q for k in keywords)`, it is extremely prone to misclassifying conversational semantic questions as structured ones.
- **Example:** *"What is the general pattern of temperature deeper in the ocean?"* -> Contains "deeper", routes to Structured. The structured engine will fail because there is no specific depth value to filter.
- **Example:** *"Explain why the Indian ocean is getting hotter over time."* -> Contains "over time", routes to Structured. The structured engine will attempt to plot a timeseries without understanding the "explain why" intent.

### 2. High False-Negative Rate (Medium Risk)
If a user asks a highly specific data question without using the exact hardcoded keywords, it routes to Semantic.
- **Example:** *"How cold does it get at 1000 meters?"* -> "meters" is not in the depth keywords, and "1000 meters" does not match the `\d+\s*m\b` regex (which expects "m", not "meters"). It routes to Semantic, which may hallucinate an answer instead of running the exact math.
- **Example:** *"Show me readings in the Atlantic."* -> "Atlantic" is not a geo keyword. "Show" is not a visualization keyword. Routes to Semantic. 

### 3. "Over Time" Conflict
"over time" is listed as a visualization keyword triggering a structured query, but the router's own comments say:
`# Semantic keywords (summarize, explain, describe, trend, pattern, overview, insight)`
Yet "trend" is ALSO in the `visualization_keywords` list!
This means asking for a "trend" or "pattern over time" forces a structured query, completely bypassing the semantic RAG pipeline which is designed for explanations and summaries.

### 4. No Hybrid Routing / Ambiguity Handling
The router treats queries as mutually exclusive. A user cannot ask a mixed query like:
*"Show me the temperature trend at 500m, and explain what ocean current causes it."*
The router immediately shunts this to the Structured engine, abandoning the explanation.

## Recommendations
1. **Implement an LLM-based Router/Classifier:** Instead of brittle keyword matching, use a fast LLM (or a fine-tuned classifier) to determine intent.
2. **Support Hybrid Routing:** Allow queries to trigger *both* paths and combine the outputs.
3. **Fix the Regex:** Expand the depth regex to support full words: `r"\d+\s*(m|meter|meters)\b"`.
4. **Remove Semantic/Structured Conflicts:** Do not overlap keywords like "trend" which clearly straddle both domains.
