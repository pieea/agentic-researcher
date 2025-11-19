from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    openai_api_key: str
    tavily_api_key: str
    database_url: str = "sqlite:///./agentic_researcher.db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_search_results: int = 30
    cache_expiry_hours: int = 24
