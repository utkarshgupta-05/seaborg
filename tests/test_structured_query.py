from router.query_router import classify_query, QueryType
from structured_query.engine import answer_structured_query


def test():
    queries = [
        "temperature at 500m",
        "temperature at 1000m",
        "Atlantic Ocean temperatures",
        "average temperature at 200m",
        "temperature at 500m in Atlantic Ocean"
    ]
    
    for q in queries:
        print(f"\n--- Query: {q} ---")
        qtype = classify_query(q)
        assert qtype == QueryType.STRUCTURED, f"Failed to classify '{q}' as STRUCTURED"
        
        result = answer_structured_query(q)
        print("\nSummary:")
        print(result["summary"])
        print("\nMetadata:")
        print(result["metadata"])
        print("\nRows shape:", result["rows"].shape)


if __name__ == "__main__":
    test()
