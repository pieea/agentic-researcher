import pytest
from src.config import Settings

def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    settings = Settings()

    assert settings.openai_api_key == "test-openai-key"
    assert settings.tavily_api_key == "test-tavily-key"
    assert settings.database_url == "sqlite:///./agentic_researcher.db"
