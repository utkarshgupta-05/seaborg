# Documentation Review

## Overview
The repository contains two primary documentation files: `README.md` and `SEABORG_AGENT_README.md`. 

## Issues & Gaps

### 1. Tutorial-Style README (Medium Risk)
The current `README.md` reads like a tutorial or prompt for an AI agent on *how to build the project from scratch* (e.g., "Create this exact structure before writing any code", Phase 1 through Phase 6 code snippets). 
- **The Gap:** It does not document the *final, actual* state of the repository. If a new developer joins the project, the README is essentially useless for understanding how to use the API endpoints, how to run tests, or how the RAG pipeline is currently configured.
- **Actionable Fix:** Rewrite the README to include standard sections: Installation, Running Locally, Architecture Diagram, API Documentation (or link to `/docs`), Environment Variables, and Testing.

### 2. Missing API Documentation (Low Risk)
FastAPI auto-generates Swagger UI (`/docs`), which is excellent. However, some of the docstrings in the Python routes are inaccurate or missing. For instance, `/api/export` claims it streams data, but it buffers the entire dataset in memory. The swagger docs will mislead consumers of the API.

### 3. Missing Architecture Explanations (Medium Risk)
Nowhere in the documentation does it explain the "Split Brain" architecture (PostgreSQL vs. Parquet) or the limitations of the Query Router. A new developer would likely assume the database is the primary source of truth, only to be confused when modifications to Postgres aren't reflected in the Chat API.

## Recommendations
1. **Archive the Tutorial README:** Move the current `README.md` to `docs/development_plan.md`.
2. **Create a standard `README.md`:** Focus on environment setup, running the server (`uvicorn api.main:app`), and a high-level system diagram.
3. **Document Environment Variables:** Clearly list required variables (`DATABASE_URL`, `GROQ_API_KEY`, `FAISS_INDEX_PATH`, etc.) and what they do.
