import pytest
from unittest.mock import Mock, patch
from src.workflow.state import ResearchState
from src.workflow.graph import create_research_workflow

def test_research_state_initialization():
    state = ResearchState(
        query="AI agents",
        status="initialized",
        raw_results=[],
        clusters=[]
    )

    assert state["query"] == "AI agents"
    assert state["status"] == "initialized"
    assert state["raw_results"] == []
    assert state["clusters"] == []

@patch("src.workflow.graph.Settings")
@patch("src.workflow.graph.SearchAgent")
@patch("src.workflow.graph.AnalysisAgent")
@patch("src.workflow.graph.InsightAgent")
def test_workflow_execution(mock_insight, mock_analysis, mock_search, mock_settings):
    # Setup settings mock
    mock_settings_instance = Mock()
    mock_settings_instance.tavily_api_key = "test-key"
    mock_settings_instance.openai_api_key = "test-key"
    mock_settings_instance.embedding_model = "test-model"
    mock_settings_instance.max_search_results = 10
    mock_settings.return_value = mock_settings_instance

    # Setup mocks
    mock_search_instance = Mock()
    mock_search_instance.search.return_value = [
        {"title": "Test", "content": "Content", "url": "http://test.com", "score": 1.0}
    ]
    mock_search.return_value = mock_search_instance

    workflow = create_research_workflow()

    initial_state = ResearchState(
        query="test query",
        status="initialized",
        raw_results=[],
        clusters=[],
        insights={}
    )

    # Note: Full execution test would require mocking all agents
    # This is a structure test
    assert workflow is not None
