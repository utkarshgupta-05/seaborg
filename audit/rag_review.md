# RAG Pipeline Review

## Overview
The Semantic Retrieval-Augmented Generation (RAG) pipeline is triggered when the Query Router defaults to a `SEMANTIC` intent. It uses Hugging Face (`all-MiniLM-L6-v2`) or a local PyTorch fallback to embed the query, performs a FAISS similarity search against an `IndexFlatL2`, retrieves the top 5 nearest Pandas rows, and injects them into an LLM prompt via Groq.

## Critical Flaws

### 1. Inappropriate Use of Embeddings for Tabular Numerical Data (Critical Risk)
The system attempts to perform RAG on structured, tabular oceanographic data by flattening every row into a sentence:
`"Float X recorded a temperature of Y°C and salinity of Z PSU at Dm depth on DATE at coordinates (LAT, LON)."`

**Why this fails:**
Embedding models (like `all-MiniLM-L6-v2`) are trained to represent the semantic, contextual meaning of language. They have absolutely no concept of mathematics, inequalities, or geographic proximity. 
If a user asks *"Find floats deeper than 1000 meters"*, the FAISS index calculates the mathematical vector distance between that query and the flattened sentences. It will return rows based on textual similarity (e.g., sentences that simply contain the word "meters" or "deeper"), **not** rows where the numerical value of `depth_m > 1000`. 
**Result:** The RAG pipeline will routinely retrieve completely irrelevant rows for any numeric, date, or geographic query.

### 2. Low "Top-K" and Context Window Starvation (High Risk)
`retriever.py` hardcodes `top_k=5`. An entire ocean contains thousands of floats. Returning only 5 floats to the LLM means the LLM has zero statistical power to identify trends, averages, or general ocean behavior. If a user asks *"What is the general temperature pattern in the Atlantic?"*, the LLM is only given 5 random floats that happened to be textually similar to the word "Atlantic" and will confidently hallucinate a "trend" based on just 5 data points.

### 3. Redundant LLM Calls (Medium Risk)
Inside `llm/query_engine.py`, the `answer_query` function generates a SQL query string using a *second* LLM call (`generate_sql(question)`) on every semantic query, just so the frontend can display it in the UI. This doubles the latency and API cost of every request for no functional benefit, since the SQL is never actually executed.

### 4. Poor Fallback Behavior (Medium Risk)
If FAISS retrieves rows, the LLM will always attempt to answer based on those rows. However, because FAISS always returns the *nearest* 5 vectors (even if they are incredibly far away in vector space), the LLM might be given data about the Pacific Ocean when the user asked about the Indian Ocean. The prompt forbids hallucination, but the LLM will just say "Based on the data, the Pacific ocean is..." which is a wrong answer to the user's question.

## Recommendations
1. **Ditch FAISS for Tabular Data:** True semantic RAG is for unstructured text (PDFs, docs). For tabular data, use an Agentic approach (Text-to-SQL or Text-to-Pandas) where the LLM writes code to query the data deterministically, rather than trying to embed numbers into a vector space.
2. **Implement Vector Distance Thresholding:** If FAISS must be kept, enforce a maximum L2 distance threshold so that completely unrelated vectors are rejected rather than blindly passing the top 5 to the LLM.
3. **Remove the redundant `generate_sql` call:** Only call it if the frontend explicitly requests the SQL string, or remove it entirely from the semantic path.
