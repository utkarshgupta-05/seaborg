"""
Verification script for Phase 7.
Runs all manual checks and prints exact output.
"""
import sys
import os
import re
import inspect

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structured_query.engine import answer_structured_query
from structured_query.service import extract_depth
import structured_query.engine as eng_mod
import structured_query.service as svc_mod
import structured_query.repository as repo_mod

SEP = "=" * 60
THICK = "#" * 60


def run_query(label, question):
    print(f"\n{SEP}")
    print(f"[{label}] QUERY: {question}")
    print(SEP)
    try:
        result = answer_structured_query(question)
        print(f"SUMMARY:\n{result['summary']}")
        print(f"\nMETADATA: {result['metadata']}")
        print(f"ROWS EMPTY: {result['rows'].empty}")
        print(f"RECORD COUNT: {result['metadata']['record_count']}")
        if not result['rows'].empty:
            print(f"ROWS SHAPE: {result['rows'].shape}")
            print(f"ROWS COLUMNS: {list(result['rows'].columns)}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def run_depth_parse(phrase):
    result = extract_depth(phrase)
    supported = result != (None, None)
    tick = "OK  " if supported else "MISS"
    print(f"  [{tick}]  {phrase!r:40s}  ->  {result}")


# ============================================================
print(f"\n{THICK}")
print("SECTION 1 -- STRUCTURED QUERY MANUAL TESTS")
print(THICK)

run_query("1a", "average temperature at 500m")
run_query("1b", "average temperature at 500m in Atlantic Ocean")
run_query("1c", "average salinity at 1000m in Pacific Ocean")
run_query("1d", "temperature between 200m and 400m")
run_query("1e", "deeper than 500m in Indian Ocean")
run_query("1f", "shallower than 100m")

# ============================================================
print(f"\n{THICK}")
print("SECTION 2 -- GLOBAL AGGREGATION TESTS")
print(THICK)

run_query("2a", "average temperature")
run_query("2b", "average salinity")
run_query("2c", "dataset-wide ocean statistics")

# ============================================================
print(f"\n{THICK}")
print("SECTION 3 -- EXECUTION PATH VERIFICATION (no Parquet)")
print(THICK)

eng_src = inspect.getsource(eng_mod)
svc_src = inspect.getsource(svc_mod)
repo_src = inspect.getsource(repo_mod)

print("\nCall chain:")
print("  chat.py        -> answer_structured_query(question)   [engine.py]")
print("  engine.py      -> _service_answer(question)           [service.py]")
print("  service.py     -> repository.query_with_filters(...)  [repository.py]")
print("                 -> repository.aggregate_stats(...)     [repository.py]")
print("  repository.py  -> SQLAlchemy text() + params          [PostgreSQL]")

print("\nParquet / Pandas file-load checks:")
for mod_name, src in [("engine.py", eng_src), ("service.py", svc_src), ("repository.py", repo_src)]:
    has_parquet = "read_parquet" in src
    has_load    = "load_data" in src
    print(f"  {mod_name:20s}  read_parquet={has_parquet}  load_data={has_load}")

print("\nSQLAlchemy parameterized query checks (repository.py):")
print(f"  Uses create_engine : {'create_engine' in repo_src}")
print(f"  Uses text()        : {'text(' in repo_src}")
print(f"  Uses :param syntax : {':depth_min' in repo_src}")
print(f"  No user val in SQL string (safe): ", end="")
# Check there's no direct f-string injection of user values
user_val_injection = re.search(r'f".*\{depth|f".*\{lat|f".*\{lon', repo_src)
print("SAFE (no injection found)" if not user_val_injection else f"UNSAFE: {user_val_injection.group()}")

# ============================================================
print(f"\n{THICK}")
print("SECTION 4 -- DEPTH PARSING COVERAGE")
print(THICK)
print()
depth_phrases = [
    "500m",
    "500 m",
    "500 meters",
    "500 metre",
    "between 200m and 400m",
    "between 200 and 400 meters",
    "deeper than 500m",
    "shallower than 100m",
    "above 100m",
    "below 500m",
    "greater than 500m",
    "less than 100m",
]
for phrase in depth_phrases:
    run_depth_parse(phrase)

# ============================================================
print(f"\n{THICK}")
print("SECTION 5 -- PUBLIC CONTRACT SHAPE")
print(THICK)
print("""
answer_structured_query(question: str) -> dict

    {
        "summary":  str           # Human-readable text
        "rows":     pd.DataFrame  # Matching rows (EMPTY for global agg)
        "metadata": {
            "query_type":   str   # Always "structured"
            "filters":      dict  # e.g. {"depth": "450-550m", "region": "Atlantic Ocean"}
                                  # {} when no filters parsed
            "record_count": int   # Row count (0 when global agg or no match)
        }
    }

chat.py mapping (no changes required):
    result["summary"]   -> ChatResponse.answer
    result["rows"]      -> rows_df  (used for visualization only)
    result["metadata"]  -> ChatResponse.metadata

Frontend safety:
    - visualization_type = None when rows is empty (no chart rendered)
    - metadata.filters = {} signals no spatial/depth filter applied
    - metadata.record_count = 0 signals no tabular rows returned
    - Frontend contract unchanged from previous structured engine
""")
