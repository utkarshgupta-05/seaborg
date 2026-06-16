import json
from fastapi.testclient import TestClient

from api.main import app
from api.models import ChatRequest

client = TestClient(app)

def run_tests():
    with TestClient(app) as client:
        print("=== STRUCTURED QUERIES ===")
        structured_queries = [
            "temperature at 500m",
            "temperature at 500m in Atlantic Ocean",
            "average temperature at 200m",
            "Atlantic Ocean temperature"
        ]
        for q in structured_queries:
            print(f"\nTesting: {q}")
            req = {"message": q}
            response = client.post("/api/chat", json=req)
            assert response.status_code == 200, f"Error: {response.text}"
            data = response.json()
            print("SQL Used:", data["sql_used"])
            print("Metadata:", data.get("metadata"))
            print("Answer preview:", data["answer"].split("\n")[0])
            assert data["sql_used"] == "N/A (Structured Engine)"

        print("\n\n=== SEMANTIC QUERIES ===")
        semantic_queries = [
            "summarize this float profile",
            "describe temperature trends",
            "explain salinity structure"
        ]
        for q in semantic_queries:
            print(f"\nTesting: {q}")
            req = {"message": q}
            response = client.post("/api/chat", json=req)
            assert response.status_code == 200, f"Error: {response.text}"
            data = response.json()
            print("SQL Used:", data["sql_used"])
            print("Metadata:", data.get("metadata"))
            print("Answer preview:", data["answer"].split("\n")[0])
            assert data["metadata"]["query_type"] == "semantic"

if __name__ == "__main__":
    run_tests()
