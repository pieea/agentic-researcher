import pytest
from unittest.mock import Mock, patch
from src.agents.insight_agent import InsightAgent
from datetime import datetime

@pytest.fixture
def mock_llm():
    with patch("src.agents.insight_agent.ChatOpenAI") as mock:
        mock_instance = Mock()
        mock_instance.invoke.return_value = Mock(
            content="1. AI agents are growing rapidly\n2. Focus on autonomous systems\n3. Integration with LLMs is key"
        )
        mock.return_value = mock_instance
        yield mock_instance

def test_insight_agent_generates_summary(mock_llm):
    agent = InsightAgent(api_key="test-key")

    clusters = [
        {"name": "Autonomous Agents", "size": 10, "keywords": ["autonomous", "agents"]},
        {"name": "LLM Integration", "size": 8, "keywords": ["llm", "integration"]}
    ]

    insights = agent.generate_insights("AI agents", clusters)

    assert "insights" in insights
    assert len(insights["insights"]) > 0
    # Check that we got some meaningful insight
    insight_text = insights["insights"][0].lower()
    assert "ai" in insight_text or "agents" in insight_text or "autonomous" in insight_text
