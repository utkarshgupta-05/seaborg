# RAG Failure Cases Validation

The SeaBorg RAG pipeline fundamentally misapplies semantic vector search to tabular, numeric data. Below is a validation of how the system fails across different query archetypes.

### Failure Case 1: Numerical Inequalities
- **Test Query:** *"Show me floats where temperature is exactly 12.5°C"*
- **Expected Behavior:** System filters the database for `temp_c = 12.5` and returns those rows.
- **Likely Failure Mode:** FAISS embedding search does not understand exact numeric matching. The vector for "12.5" is not mathematically closest to the vector for the string "12.5". It might return 12.1°C, 2.5°C, or entirely unrelated temperatures.
- **Severity:** Critical
- **Recommended Fix:** Route all queries containing numbers to the Structured Engine (or Text-to-SQL).

### Failure Case 2: Geographic Bounding Boxes
- **Test Query:** *"What is the salinity at coordinates 10.5N, 45.2E?"*
- **Expected Behavior:** System filters for latitude/longitude ranges near those coordinates.
- **Likely Failure Mode:** The embedding model tokenizes "10.5N" and "45.2E" as text tokens. It searches for textual similarity in the FAISS index. Floats at 10.5S might be retrieved because "10.5" matches textually, entirely ignoring geographic reality.
- **Severity:** Critical
- **Recommended Fix:** Extract coordinates using an LLM and pass them as strict filters to Pandas/PostgreSQL before using RAG.

### Failure Case 3: Macro-level Aggregations
- **Test Query:** *"What is the average temperature of the Pacific Ocean?"*
- **Expected Behavior:** A mean calculation over all millions of Pacific Ocean rows.
- **Likely Failure Mode:** The router defaults to Semantic because "average" might be misspelled or missing. `top_k=5` pulls 5 random rows that textually mention "Pacific". The LLM averages those 5 rows and returns a highly inaccurate response that the user assumes represents the whole ocean.
- **Severity:** High
- **Recommended Fix:** Always execute mathematical aggregations via SQL/Pandas, never via LLM summarization of a top-K sample.

### Failure Case 4: Complete Miss / Empty Scenarios
- **Test Query:** *"Tell me about floats in the Arctic Ocean"* (Assuming no Arctic data exists).
- **Expected Behavior:** "No data found for the Arctic Ocean."
- **Likely Failure Mode:** FAISS *always* returns the 5 nearest vectors, regardless of distance. It will retrieve 5 floats from the Atlantic Ocean (as it's the closest textual match). The LLM will then answer using Atlantic data, confusing the user.
- **Severity:** High
- **Recommended Fix:** Implement an L2 distance threshold in `retriever.py` to discard results that are not semantically close to the query.
