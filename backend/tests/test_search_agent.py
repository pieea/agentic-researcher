import pytest
from unittest.mock import Mock, patch
from src.agents.search_agent import SearchAgent
from datetime import datetime

@pytest.fixture
def mock_tavily_client():
    with patch("src.agents.search_agent.TavilyClient") as mock:
        mock_instance = Mock()
        mock_instance.search.return_value = {
            "results": [
                {
                    "title": "AI Agents Overview",
                    "url": "https://example.com/ai-agents",
                    "content": "AI agents are autonomous systems...",
                    "score": 0.95,
                    "published_date": "2025-11-15T10:00:00Z"
                }
            ]
        }
        mock.return_value = mock_instance
        yield mock_instance

def test_search_agent_executes_search(mock_tavily_client):
    agent = SearchAgent(api_key="test-key")
    results = agent.search("AI agents", max_results=10)

    assert len(results) == 1
    assert results[0]["title"] == "AI Agents Overview"
    assert results[0]["url"] == "https://example.com/ai-agents"
    assert "published_date" in results[0]
