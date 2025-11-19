import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_create_research_request():
    response = client.post(
        "/api/research",
        json={"query": "AI agents"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["status"] in ["processing", "queued"]
