# Scalability Review

## Overview
This document evaluates the system's ability to scale based on dataset size and concurrent user load.

### Scenario 1: Current Dataset (~30,000 records)
- **Memory Footprint:** The Parquet file is small (~5-10MB). Loading it twice into Pandas takes ~20MB. FAISS index takes < 50MB.
- **API Performance:** Fast enough, though single-threaded blocking means it can only serve ~1 request per 5 seconds concurrently.
- **Verdict:** Passes on Render Free Tier (512MB RAM), but only for single users.

### Scenario 2: 10x Growth (~300,000 records)
- **Memory Footprint:** Pandas will consume ~200MB. FAISS index will consume ~400MB.
- **API Performance:** Pandas filtering (Structured Query) will still be near-instant. FAISS L2 search will slow down slightly but remain sub-second.
- **Verdict:** Fails on Render Free Tier due to exceeding 512MB RAM. Would survive on a standard 1GB/2GB VPS. The Postgres `COUNT(DISTINCT)` stats query will start taking ~500ms.

### Scenario 3: 100x Growth (~3,000,000 records)
- **Memory Footprint:** Pandas will consume ~2GB. FAISS index will consume ~4GB.
- **API Performance:** `engine.py` calls `_df[mask].copy()`. Deep copying a 1GB dataframe for every structured query will trigger massive garbage collection pauses and OOM kills on standard servers.
- **Verdict:** Total System Failure. The "load everything into Pandas" architecture collapses here. Postgres export queries will timeout.

### Scenario 4: 1M+ to 10M+ records
- **Memory Footprint:** 10GB+ RAM required just to boot the application.
- **FAISS Bottlenecks:** `IndexFlatL2` does an exact exhaustive search. At 10M records, `faiss.search()` on CPU will take several seconds per query, blocking the server entirely.
- **Verdict:** Unusable.

## The First Failure Point
**Memory Exhaustion via Pandas Duplication:** The system will crash on boot (or during the first structured query deep copy) as soon as the dataset reaches around ~500,000 records on a memory-constrained server.

## Remediation Plan for Scale
1. **Abandon Pandas for Queries:** Move all structured querying directly to PostgreSQL. PostgreSQL is designed to handle millions of rows efficiently via indexes.
2. **Move FAISS out of memory:** If semantic search is kept, move to a dedicated vector database (Pinecone, Qdrant, or pgvector) so the FastAPI server remains stateless and doesn't need to hold the index in RAM.
3. **Use HNSW for FAISS:** If sticking with FAISS, switch from `IndexFlatL2` to `IndexHNSWFlat` to avoid exhaustive linear search times at scale.
