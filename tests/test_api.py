import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_chat_endpoint_semantic(client):
    response = client.post(
        "/api/chat",
        json={"message": "What is the ocean temperature?"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields exist
    for field in ["answer", "chart_type", "float_ids", "sql_used", "confidence"]:
        assert field in data
        
    # chart_type must be in valid values
    assert data["chart_type"] in ["map", "profile", "timeseries", "none"]
    
    # Non-visualization query should have None in new fields
    assert data.get("visualization_type") is None
    assert data.get("visualization_data") is None

def test_chat_endpoint_sql_safety(client):
    response = client.post(
        "/api/chat",
        json={"message": "What is the ocean temperature? DROP TABLE argo_profiles;"}
    )
    assert response.status_code == 200
    data = response.json()
    sql_used = data.get("sql_used")
    if sql_used:
        assert "DROP TABLE" not in sql_used

def test_floats_endpoint(client):
    response = client.get("/api/floats")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "floats" in data
    assert isinstance(data["floats"], list)

def test_chat_visualization_query(client):
    # Test show temperature profile (profile chart)
    response = client.post(
        "/api/chat",
        json={"message": "show temperature profile"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["visualization_type"] == "profile"
    assert data["visualization_data"] is not None
    assert "data" in data["visualization_data"]
    assert "layout" in data["visualization_data"]
    assert "Temperature Profile" in data["chart_title"]
    assert data["chart_description"] is not None

    # Test show float location (map chart)
    response_map = client.post(
        "/api/chat",
        json={"message": "show float location"}
    )
    assert response_map.status_code == 200
    data_map = response_map.json()
    assert data_map["visualization_type"] == "map"
    assert data_map["visualization_data"] is not None
    assert "ARGO Float Positions" in data_map["chart_title"]
