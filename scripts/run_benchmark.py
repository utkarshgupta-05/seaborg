import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load FAISS index before running the benchmark loop
from rag.retriever import load_index
try:
    load_index()
except Exception as e:
    print(f"Warning: Failed to load FAISS index: {e}")

from api.models import ChatRequest
from api.routes.chat import chat
from router.query_router import classify_query

def main():
    benchmark_file = PROJECT_ROOT / "benchmark" / "queries.json"
    if not benchmark_file.exists():
        print(f"Error: Benchmark file not found at {benchmark_file}")
        sys.exit(1)

    with open(benchmark_file, "r") as f:
        queries = json.load(f)

    print(f"Running SeaBorg Benchmark ({len(queries)} queries)...\n")
    print(f"{'ID':<3} | {'Expected':<10} | {'Actual':<10} | {'Rows':<5} | {'Time':<6} | {'Result':<6} | Query")
    print("-" * 115)
    
    passed = 0

    for q in queries:
        req = ChatRequest(message=q["query"])
        start_t = time.time()
        
        try:
            # We bypass HTTP and directly invoke the chat router for speed
            resp = chat(req)
            elapsed = time.time() - start_t
            
            # The routing logic classifies the intent deterministically
            actual_route = classify_query(q["query"]).value
            
            # Heuristic to determine if data was grounded and matched
            has_data = len(resp.float_ids) > 0 if resp.float_ids else False
            
            is_pass = (actual_route == q["expected_route"])
            if is_pass:
                passed += 1
                
            status = "PASS" if is_pass else "FAIL"
            rows_str = str(len(resp.float_ids)) if has_data else "0"
            
            # Print a truncated version of the query to keep output clean
            short_query = (q["query"][:45] + "...") if len(q["query"]) > 48 else q["query"]
            
            print(f"{q['id']:<3} | {q['expected_route']:<10} | {actual_route:<10} | {rows_str:<5} | {elapsed:>4.1f}s | {status:<6} | {short_query}")
            
        except Exception as e:
            print(f"{q['id']:<3} | ERROR: {e}")

    print("-" * 115)
    print(f"Benchmark Complete. Score: {passed}/{len(queries)} ({(passed/len(queries))*100:.1f}%)")

if __name__ == "__main__":
    main()
