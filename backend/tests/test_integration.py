import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from src.main import app

client = TestClient(app)

@pytest.mark.integration
@patch("src.workflow.graph.SearchAgent")
@patch("src.workflow.graph.AnalysisAgent")
@patch("src.workflow.graph.InsightAgent")
def test_full_research_workflow(mock_insight, mock_analysis, mock_search):
    """Test the complete research workflow end-to-end."""

    # Setup mocks
    mock_search_instance = Mock()
    mock_search_instance.search.return_value = [
        {
            "title": "AI Agents Overview",
            "content": "AI agents are autonomous systems...",
            "url": "https://example.com/1",
            "score": 0.95,
            "published_date": "2025-11-15T10:00:00Z",
            "source": "example.com"
        }
    ]
    mock_search.return_value = mock_search_instance

    # Create research request
    response = client.post(
        "/api/research",
        json={"query": "AI agents"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data

    request_id = data["request_id"]

    # TODO: Wait for workflow completion and verify results
    # In production, use async testing or background task handling
