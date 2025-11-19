# Market Research Platform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI-powered market research platform that analyzes trends using Tavily search, LangGraph workflows, and automatic topic clustering.

**Architecture:** Multi-agent LangGraph workflow (Search → Analysis → Insight agents) orchestrated by FastAPI backend, with React frontend consuming SSE streams for real-time progress updates. SQLite for caching results.

**Tech Stack:** FastAPI, LangGraph, Tavily API, SQLite, SQLAlchemy, sentence-transformers, scikit-learn (HDBSCAN), React, TypeScript, Recharts

---

## Phase 1: Project Setup & Backend Foundation

### Task 1: Initialize Python Backend Structure

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/__init__.py`
- Create: `backend/src/main.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/.env.example`

**Step 1: Create pyproject.toml with dependencies**

```toml
[tool.poetry]
name = "agentic-researcher-backend"
version = "0.1.0"
description = "AI-powered market research platform backend"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
sqlalchemy = "^2.0.0"
langgraph = "^0.0.20"
langchain = "^0.1.0"
langchain-openai = "^0.0.2"
tavily-python = "^0.3.0"
sentence-transformers = "^2.2.2"
scikit-learn = "^1.3.0"
hdbscan = "^0.8.33"
python-dotenv = "^1.0.0"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
httpx = "^0.25.0"
black = "^23.11.0"
ruff = "^0.1.6"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**Step 2: Create main.py with FastAPI app**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Agentic Researcher API",
    description="AI-powered market research platform",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 3: Create .env.example**

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=sqlite:///./agentic_researcher.db
```

**Step 4: Install dependencies**

Run: `cd backend && poetry install`
Expected: Dependencies installed successfully

**Step 5: Test the server starts**

Run: `cd backend && poetry run uvicorn src.main:app --reload`
Expected: Server starts on http://127.0.0.1:8000

Stop the server (Ctrl+C)

**Step 6: Commit**

```bash
git add backend/
git commit -m "feat: initialize Python backend with FastAPI

Setup project structure with Poetry, FastAPI, and core dependencies"
```

---

### Task 2: Database Models and Schema

**Files:**
- Create: `backend/src/database.py`
- Create: `backend/src/models.py`
- Create: `backend/tests/test_models.py`

**Step 1: Write test for database connection**

Create `backend/tests/test_models.py`:

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database import Base, get_db
from src.models import ResearchQuery, SearchResult, Cluster, ClusterMembership

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

def test_create_research_query(test_db):
    query = ResearchQuery(query="AI agent", status="processing")
    test_db.add(query)
    test_db.commit()

    result = test_db.query(ResearchQuery).filter_by(query="AI agent").first()
    assert result is not None
    assert result.query == "AI agent"
    assert result.status == "processing"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_models.py -v`
Expected: FAIL with import errors (modules don't exist)

**Step 3: Create database.py**

Create `backend/src/database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agentic_researcher.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Step 4: Create models.py**

Create `backend/src/models.py`:

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from src.database import Base

class ResearchQuery(Base):
    __tablename__ = "research_queries"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(hours=24), nullable=False)
    status = Column(String, default="processing", nullable=False)  # processing, completed, failed

    search_results = relationship("SearchResult", back_populates="query")
    clusters = relationship("Cluster", back_populates="query")

class SearchResult(Base):
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("research_queries.id"), nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    summary = Column(Text, nullable=False)
    published_date = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    relevance_score = Column(Float, default=1.0, nullable=False)

    query = relationship("ResearchQuery", back_populates="search_results")
    cluster_memberships = relationship("ClusterMembership", back_populates="result")

class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("research_queries.id"), nullable=False)
    cluster_name = Column(String, nullable=False)
    size = Column(Integer, default=0, nullable=False)
    keywords = Column(JSON, nullable=False)

    query = relationship("ResearchQuery", back_populates="clusters")
    memberships = relationship("ClusterMembership", back_populates="cluster")

class ClusterMembership(Base):
    __tablename__ = "cluster_memberships"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    result_id = Column(Integer, ForeignKey("search_results.id"), nullable=False)
    distance = Column(Float, nullable=False)

    cluster = relationship("Cluster", back_populates="memberships")
    result = relationship("SearchResult", back_populates="cluster_memberships")
```

**Step 5: Run test to verify it passes**

Run: `cd backend && poetry run pytest tests/test_models.py -v`
Expected: PASS (1 test)

**Step 6: Commit**

```bash
git add backend/src/database.py backend/src/models.py backend/tests/test_models.py
git commit -m "feat: add SQLAlchemy models for research queries and results

Implemented ResearchQuery, SearchResult, Cluster, and ClusterMembership models"
```

---

### Task 3: Configuration and Settings

**Files:**
- Create: `backend/src/config.py`
- Create: `backend/tests/test_config.py`

**Step 1: Write test for configuration**

Create `backend/tests/test_config.py`:

```python
import pytest
from src.config import Settings

def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    settings = Settings()

    assert settings.openai_api_key == "test-openai-key"
    assert settings.tavily_api_key == "test-tavily-key"
    assert settings.database_url == "sqlite:///./agentic_researcher.db"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_config.py -v`
Expected: FAIL (Settings class doesn't exist)

**Step 3: Create config.py**

Create `backend/src/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str
    database_url: str = "sqlite:///./agentic_researcher.db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_search_results: int = 30
    cache_expiry_hours: int = 24

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**Step 4: Run test to verify it passes**

Run: `cd backend && poetry run pytest tests/test_config.py -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add backend/src/config.py backend/tests/test_config.py
git commit -m "feat: add configuration management with pydantic-settings"
```

---

## Phase 2: LangGraph Workflow Implementation

### Task 4: Search Agent with Tavily Integration

**Files:**
- Create: `backend/src/agents/__init__.py`
- Create: `backend/src/agents/search_agent.py`
- Create: `backend/tests/test_search_agent.py`

**Step 1: Write test for search agent**

Create `backend/tests/test_search_agent.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_search_agent.py -v`
Expected: FAIL (SearchAgent doesn't exist)

**Step 3: Create search_agent.py**

Create `backend/src/agents/search_agent.py`:

```python
from tavily import TavilyClient
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SearchAgent:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        """
        Execute search via Tavily API with recency weighting.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, content, score, published_date
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_domains=[],
                exclude_domains=[]
            )

            results = []
            for item in response.get("results", []):
                # Apply recency weighting
                published_date = item.get("published_date")
                recency_boost = self._calculate_recency_boost(published_date)

                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0) * recency_boost,
                    "published_date": published_date,
                    "source": self._extract_domain(item.get("url", ""))
                })

            # Sort by adjusted score
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Search completed: {len(results)} results for query '{query}'")
            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            raise

    def _calculate_recency_boost(self, published_date: str) -> float:
        """Apply higher weight to recent content (last 7 days)."""
        if not published_date:
            return 1.0

        try:
            pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            days_ago = (datetime.now(pub_date.tzinfo) - pub_date).days

            if days_ago <= 2:
                return 1.5  # Last 2 days: 50% boost
            elif days_ago <= 7:
                return 1.2  # Last week: 20% boost
            else:
                return 1.0  # Older: no boost
        except:
            return 1.0

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
```

**Step 4: Run test to verify it passes**

Run: `cd backend && poetry run pytest tests/test_search_agent.py -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add backend/src/agents/ backend/tests/test_search_agent.py
git commit -m "feat: implement SearchAgent with Tavily integration

Added recency weighting for recent content (last 7 days)"
```

---

### Task 5: Analysis Agent with Embeddings and Clustering

**Files:**
- Create: `backend/src/agents/analysis_agent.py`
- Create: `backend/tests/test_analysis_agent.py`

**Step 1: Write test for analysis agent**

Create `backend/tests/test_analysis_agent.py`:

```python
import pytest
from src.agents.analysis_agent import AnalysisAgent
import numpy as np

def test_analysis_agent_generates_embeddings():
    agent = AnalysisAgent()

    texts = [
        "AI agents are autonomous systems",
        "Machine learning models for prediction",
        "Autonomous agents in robotics"
    ]

    embeddings = agent.generate_embeddings(texts)

    assert len(embeddings) == 3
    assert embeddings.shape[1] > 0  # Has embedding dimensions

def test_analysis_agent_clusters_documents():
    agent = AnalysisAgent()

    # Create mock embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # Cluster 1
        [1.1, 0.1, 0.0],  # Cluster 1
        [0.0, 1.0, 0.0],  # Cluster 2
        [0.1, 1.1, 0.0],  # Cluster 2
    ])

    labels = agent.cluster_embeddings(embeddings, min_cluster_size=2)

    assert len(labels) == 4
    assert len(set(labels)) >= 2  # At least 2 clusters
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_analysis_agent.py -v`
Expected: FAIL (AnalysisAgent doesn't exist)

**Step 3: Create analysis_agent.py**

Create `backend/src/agents/analysis_agent.py`:

```python
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings (shape: [n_texts, embedding_dim])
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 3
    ) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.

        Args:
            embeddings: NumPy array of embeddings
            min_cluster_size: Minimum cluster size for HDBSCAN

        Returns:
            NumPy array of cluster labels (-1 for noise)
        """
        logger.info(f"Clustering {len(embeddings)} embeddings")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        logger.info(f"Found {n_clusters} clusters (noise: {sum(labels == -1)})")
        return labels

    def extract_cluster_keywords(
        self,
        texts: List[str],
        labels: np.ndarray,
        top_k: int = 5
    ) -> Dict[int, List[str]]:
        """
        Extract top keywords for each cluster using TF-IDF.

        Args:
            texts: Original text documents
            labels: Cluster labels for each document
            top_k: Number of top keywords to extract per cluster

        Returns:
            Dictionary mapping cluster_id -> list of keywords
        """
        cluster_keywords = {}
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            # Get texts for this cluster
            cluster_texts = [text for i, text in enumerate(texts) if labels[i] == label]

            if not cluster_texts:
                continue

            # Extract keywords using TF-IDF
            try:
                vectorizer = TfidfVectorizer(max_features=top_k, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                keywords = vectorizer.get_feature_names_out()
                cluster_keywords[int(label)] = list(keywords)
            except:
                cluster_keywords[int(label)] = []

        return cluster_keywords
```

**Step 4: Run test to verify it passes**

Run: `cd backend && poetry run pytest tests/test_analysis_agent.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add backend/src/agents/analysis_agent.py backend/tests/test_analysis_agent.py
git commit -m "feat: implement AnalysisAgent with embeddings and clustering

Added sentence-transformers for embeddings and HDBSCAN for clustering"
```

---

### Task 6: Insight Agent with LLM

**Files:**
- Create: `backend/src/agents/insight_agent.py`
- Create: `backend/tests/test_insight_agent.py`

**Step 1: Write test for insight agent**

Create `backend/tests/test_insight_agent.py`:

```python
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
    assert "AI agents" in insights["insights"][0].lower() or "autonomous" in insights["insights"][0].lower()
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_insight_agent.py -v`
Expected: FAIL (InsightAgent doesn't exist)

**Step 3: Create insight_agent.py**

Create `backend/src/agents/insight_agent.py`:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class InsightAgent:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7
        )
        logger.info(f"Initialized InsightAgent with model: {model}")

    def generate_insights(
        self,
        query: str,
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate key insights from clustered research results.

        Args:
            query: Original search query
            clusters: List of cluster information (name, size, keywords)

        Returns:
            Dictionary with insights, trends, and summary
        """
        logger.info(f"Generating insights for query: {query}")

        # Prepare cluster summary
        cluster_summary = "\n".join([
            f"- {c['name']}: {c['size']} documents, keywords: {', '.join(c['keywords'][:5])}"
            for c in clusters
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert market research analyst.
            Analyze the following research clusters and provide 3-5 key insights about the topic.
            Focus on trends, patterns, and actionable takeaways.
            Be concise and specific."""),
            ("user", """Query: {query}

Clusters found:
{clusters}

Provide 3-5 key insights as a numbered list.""")
        ])

        try:
            response = self.llm.invoke(
                prompt.format_messages(query=query, clusters=cluster_summary)
            )

            # Parse insights from response
            insights_text = response.content
            insights = [
                line.strip()
                for line in insights_text.split('\n')
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))
            ]

            return {
                "insights": insights,
                "summary": insights_text,
                "cluster_count": len(clusters),
                "total_documents": sum(c['size'] for c in clusters)
            }

        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            # Fallback to basic statistics
            return {
                "insights": [
                    f"Found {len(clusters)} major topics related to '{query}'",
                    f"Analyzed {sum(c['size'] for c in clusters)} documents total",
                    f"Largest topic: {max(clusters, key=lambda x: x['size'])['name']}"
                ],
                "summary": "Basic statistical summary (LLM unavailable)",
                "cluster_count": len(clusters),
                "total_documents": sum(c['size'] for c in clusters)
            }

    def generate_cluster_names(
        self,
        keywords_list: List[List[str]]
    ) -> List[str]:
        """
        Generate meaningful names for clusters based on keywords.

        Args:
            keywords_list: List of keyword lists, one per cluster

        Returns:
            List of generated cluster names
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a naming expert. Create concise, descriptive names (2-4 words) for topics based on keywords."),
            ("user", "Keywords: {keywords}\n\nGenerate a concise topic name:")
        ])

        names = []
        for keywords in keywords_list:
            try:
                response = self.llm.invoke(
                    prompt.format_messages(keywords=", ".join(keywords))
                )
                name = response.content.strip().strip('"').strip("'")
                names.append(name)
            except:
                # Fallback to simple name
                names.append(f"Topic: {keywords[0] if keywords else 'Unknown'}")

        return names
```

**Step 4: Run test to verify it passes**

Run: `cd backend && poetry run pytest tests/test_insight_agent.py -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add backend/src/agents/insight_agent.py backend/tests/test_insight_agent.py
git commit -m "feat: implement InsightAgent with LLM-based analysis

Added cluster naming and insight generation using GPT-4"
```

---

### Task 7: LangGraph Workflow Orchestration

**Files:**
- Create: `backend/src/workflow/__init__.py`
- Create: `backend/src/workflow/state.py`
- Create: `backend/src/workflow/graph.py`
- Create: `backend/tests/test_workflow.py`

**Step 1: Write test for workflow state**

Create `backend/tests/test_workflow.py`:

```python
import pytest
from src.workflow.state import ResearchState

def test_research_state_initialization():
    state = ResearchState(
        query="AI agents",
        status="initialized"
    )

    assert state["query"] == "AI agents"
    assert state["status"] == "initialized"
    assert state["raw_results"] == []
    assert state["clusters"] == []
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_workflow.py -v`
Expected: FAIL (ResearchState doesn't exist)

**Step 3: Create state.py**

Create `backend/src/workflow/state.py`:

```python
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np

class ResearchState(TypedDict, total=False):
    """State for the research workflow."""
    query: str
    raw_results: List[Dict[str, Any]]
    embeddings: Optional[np.ndarray]
    cluster_labels: Optional[np.ndarray]
    clusters: List[Dict[str, Any]]
    insights: Dict[str, Any]
    visualization_data: Dict[str, Any]
    status: str  # "searching", "analyzing", "clustering", "generating_insights", "completed", "failed"
    error: Optional[str]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && poetry run pytest tests/test_workflow.py -v`
Expected: PASS (1 test)

**Step 5: Create workflow graph**

Create `backend/src/workflow/graph.py`:

```python
from langgraph.graph import StateGraph, END
from src.workflow.state import ResearchState
from src.agents.search_agent import SearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.insight_agent import InsightAgent
from src.config import settings
import logging

logger = logging.getLogger(__name__)

def create_research_workflow() -> StateGraph:
    """Create the LangGraph workflow for market research."""

    # Initialize agents
    search_agent = SearchAgent(api_key=settings.tavily_api_key)
    analysis_agent = AnalysisAgent(model_name=settings.embedding_model)
    insight_agent = InsightAgent(api_key=settings.openai_api_key)

    # Define node functions
    def search_node(state: ResearchState) -> ResearchState:
        """Execute search via Tavily."""
        logger.info(f"Search node: querying '{state['query']}'")
        state["status"] = "searching"

        try:
            results = search_agent.search(
                state["query"],
                max_results=settings.max_search_results
            )
            state["raw_results"] = results
            state["status"] = "search_completed"
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            state["status"] = "failed"
            state["error"] = str(e)

        return state

    def analysis_node(state: ResearchState) -> ResearchState:
        """Generate embeddings and cluster results."""
        logger.info("Analysis node: generating embeddings")
        state["status"] = "analyzing"

        try:
            # Combine title and content for embedding
            texts = [
                f"{r['title']}. {r['content']}"
                for r in state["raw_results"]
            ]

            # Generate embeddings
            embeddings = analysis_agent.generate_embeddings(texts)
            state["embeddings"] = embeddings

            # Cluster embeddings
            if len(embeddings) >= 5:  # Need minimum documents for clustering
                labels = analysis_agent.cluster_embeddings(embeddings, min_cluster_size=3)
                state["cluster_labels"] = labels

                # Extract keywords per cluster
                keywords_map = analysis_agent.extract_cluster_keywords(texts, labels, top_k=5)

                # Generate cluster names
                keywords_list = [keywords_map.get(i, []) for i in range(max(labels) + 1)]
                cluster_names = insight_agent.generate_cluster_names(keywords_list)

                # Build cluster info
                clusters = []
                for cluster_id in range(max(labels) + 1):
                    cluster_docs = [
                        state["raw_results"][i]
                        for i, label in enumerate(labels)
                        if label == cluster_id
                    ]

                    clusters.append({
                        "id": cluster_id,
                        "name": cluster_names[cluster_id] if cluster_id < len(cluster_names) else f"Topic {cluster_id}",
                        "size": len(cluster_docs),
                        "keywords": keywords_map.get(cluster_id, []),
                        "documents": cluster_docs[:3]  # Top 3 representative docs
                    })

                state["clusters"] = clusters
                state["status"] = "clustering_completed"
            else:
                # Too few results, skip clustering
                state["clusters"] = [{
                    "id": 0,
                    "name": state["query"],
                    "size": len(state["raw_results"]),
                    "keywords": [],
                    "documents": state["raw_results"][:3]
                }]
                state["status"] = "clustering_skipped"

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            state["status"] = "failed"
            state["error"] = str(e)

        return state

    def insight_node(state: ResearchState) -> ResearchState:
        """Generate insights from clusters."""
        logger.info("Insight node: generating insights")
        state["status"] = "generating_insights"

        try:
            insights = insight_agent.generate_insights(
                state["query"],
                state["clusters"]
            )
            state["insights"] = insights
            state["status"] = "completed"
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            state["status"] = "failed"
            state["error"] = str(e)

        return state

    # Build graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("insight", insight_node)

    # Define edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "analysis")
    workflow.add_edge("analysis", "insight")
    workflow.add_edge("insight", END)

    return workflow.compile()
```

**Step 6: Add workflow test**

Add to `backend/tests/test_workflow.py`:

```python
from unittest.mock import Mock, patch
from src.workflow.graph import create_research_workflow

@patch("src.workflow.graph.SearchAgent")
@patch("src.workflow.graph.AnalysisAgent")
@patch("src.workflow.graph.InsightAgent")
def test_workflow_execution(mock_insight, mock_analysis, mock_search):
    # Setup mocks
    mock_search_instance = Mock()
    mock_search_instance.search.return_value = [
        {"title": "Test", "content": "Content", "url": "http://test.com", "score": 1.0}
    ]
    mock_search.return_value = mock_search_instance

    workflow = create_research_workflow()

    initial_state = ResearchState(
        query="test query",
        status="initialized"
    )

    # Note: Full execution test would require mocking all agents
    # This is a structure test
    assert workflow is not None
```

**Step 7: Run tests to verify they pass**

Run: `cd backend && poetry run pytest tests/test_workflow.py -v`
Expected: PASS (2 tests)

**Step 8: Commit**

```bash
git add backend/src/workflow/ backend/tests/test_workflow.py
git commit -m "feat: implement LangGraph workflow orchestration

Created multi-agent workflow with search, analysis, and insight nodes"
```

---

## Phase 3: FastAPI Endpoints & SSE Streaming

### Task 8: Research API Endpoints

**Files:**
- Create: `backend/src/routers/__init__.py`
- Create: `backend/src/routers/research.py`
- Create: `backend/src/schemas.py`
- Modify: `backend/src/main.py`
- Create: `backend/tests/test_api.py`

**Step 1: Write test for research endpoint**

Create `backend/tests/test_api.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd backend && poetry run pytest tests/test_api.py::test_create_research_request -v`
Expected: FAIL (endpoint doesn't exist)

**Step 3: Create schemas.py**

Create `backend/src/schemas.py`:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)

class ResearchResponse(BaseModel):
    request_id: str
    status: str

class ClusterInfo(BaseModel):
    id: int
    name: str
    size: int
    keywords: List[str]
    documents: List[Dict[str, Any]]

class InsightInfo(BaseModel):
    insights: List[str]
    summary: str
    cluster_count: int
    total_documents: int

class ResearchResult(BaseModel):
    request_id: str
    query: str
    status: str
    clusters: List[ClusterInfo]
    insights: Optional[InsightInfo]
    created_at: datetime
    completed_at: Optional[datetime]
```

**Step 4: Create research router**

Create `backend/src/routers/research.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from src.database import get_db
from src.schemas import ResearchRequest, ResearchResponse, ResearchResult
from src.workflow.graph import create_research_workflow
from src.workflow.state import ResearchState
from src.models import ResearchQuery as DBResearchQuery
import uuid
import json
import asyncio
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

# In-memory storage for active workflows (use Redis in production)
active_workflows = {}

@router.post("", response_model=ResearchResponse)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new research request."""

    # Create database record
    request_id = str(uuid.uuid4())
    db_query = DBResearchQuery(
        id=request_id,
        query=request.query,
        status="processing"
    )
    db.add(db_query)
    db.commit()

    # Start workflow in background
    background_tasks.add_task(execute_workflow, request_id, request.query, db)

    return ResearchResponse(
        request_id=request_id,
        status="processing"
    )

async def execute_workflow(request_id: str, query: str, db: Session):
    """Execute the research workflow."""
    try:
        workflow = create_research_workflow()

        initial_state = ResearchState(
            query=query,
            status="initialized",
            raw_results=[],
            clusters=[],
            insights={}
        )

        # Store initial state
        active_workflows[request_id] = initial_state

        # Execute workflow (blocking - in production use async or Celery)
        final_state = workflow.invoke(initial_state)

        # Update database
        db_query = db.query(DBResearchQuery).filter_by(id=request_id).first()
        if db_query:
            db_query.status = final_state.get("status", "completed")
            db.commit()

        # Store final state
        active_workflows[request_id] = final_state

    except Exception as e:
        logger.error(f"Workflow execution failed for {request_id}: {str(e)}")

        # Update database
        db_query = db.query(DBResearchQuery).filter_by(id=request_id).first()
        if db_query:
            db_query.status = "failed"
            db.commit()

@router.get("/{request_id}/stream")
async def stream_progress(request_id: str):
    """Stream progress updates via SSE."""

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for progress updates."""

        while True:
            if request_id not in active_workflows:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            state = active_workflows[request_id]
            status = state.get("status", "unknown")

            yield f"data: {json.dumps({'status': status})}\n\n"

            if status in ["completed", "failed"]:
                break

            await asyncio.sleep(1)  # Poll every second

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@router.get("/{request_id}")
async def get_research_result(request_id: str, db: Session = Depends(get_db)):
    """Get final research results."""

    db_query = db.query(DBResearchQuery).filter_by(id=request_id).first()

    if not db_query:
        raise HTTPException(status_code=404, detail="Research request not found")

    if request_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Results not available")

    state = active_workflows[request_id]

    return {
        "request_id": request_id,
        "query": state.get("query"),
        "status": state.get("status"),
        "clusters": state.get("clusters", []),
        "insights": state.get("insights", {}),
        "created_at": db_query.created_at,
        "completed_at": None  # TODO: track completion time
    }
```

**Step 5: Update main.py to include router**

Modify `backend/src/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import research
from src.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Agentic Researcher API",
    description="AI-powered market research platform",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 6: Run tests to verify they pass**

Run: `cd backend && poetry run pytest tests/test_api.py -v`
Expected: PASS (2 tests)

**Step 7: Commit**

```bash
git add backend/src/routers/ backend/src/schemas.py backend/src/main.py backend/tests/test_api.py
git commit -m "feat: add research API endpoints with SSE streaming

Implemented POST /api/research, GET /api/research/{id}, and SSE streaming"
```

---

## Phase 4: Frontend Implementation

### Task 9: React + TypeScript Setup

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.tsx`
- Create: `frontend/public/index.html`

**Step 1: Create package.json**

Create `frontend/package.json`:

```json
{
  "name": "agentic-researcher-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "recharts": "^2.10.0",
    "d3": "^7.8.5",
    "@types/d3": "^7.4.3",
    "typescript": "^5.3.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.0.0"
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }
}
```

**Step 2: Create tsconfig.json**

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

**Step 3: Create vite.config.ts**

Create `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

**Step 4: Create basic React app structure**

Create `frontend/src/main.tsx`:

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

Create `frontend/src/App.tsx`:

```typescript
import React, { useState } from 'react'

function App() {
  const [query, setQuery] = useState('')

  const handleSearch = async () => {
    console.log('Searching for:', query)
  }

  return (
    <div className="app">
      <header>
        <h1>Market Research Platform</h1>
      </header>

      <main>
        <div className="search-box">
          <input
            type="text"
            placeholder="Enter topic to research..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button onClick={handleSearch}>Research</button>
        </div>
      </main>
    </div>
  )
}

export default App
```

Create `frontend/src/index.css`:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: #f5f5f5;
}

.app {
  min-height: 100vh;
  padding: 2rem;
}

header {
  text-align: center;
  margin-bottom: 3rem;
}

h1 {
  color: #333;
  font-size: 2.5rem;
}

.search-box {
  max-width: 600px;
  margin: 0 auto;
  display: flex;
  gap: 1rem;
}

input {
  flex: 1;
  padding: 1rem;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
}

button {
  padding: 1rem 2rem;
  font-size: 1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}

button:hover {
  background: #0056b3;
}
```

Create `frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Market Research Platform</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

**Step 5: Install frontend dependencies**

Run: `cd frontend && npm install`
Expected: Dependencies installed

**Step 6: Test frontend runs**

Run: `cd frontend && npm run dev`
Expected: Vite server starts on http://localhost:3000

Stop the server (Ctrl+C)

**Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: initialize React + TypeScript frontend with Vite

Basic search UI with proxy to backend API"
```

---

### Task 10: API Integration & SSE Streaming

**Files:**
- Create: `frontend/src/api/research.ts`
- Create: `frontend/src/types.ts`
- Modify: `frontend/src/App.tsx`

**Step 1: Create types**

Create `frontend/src/types.ts`:

```typescript
export interface ResearchRequest {
  query: string
}

export interface ResearchResponse {
  request_id: string
  status: string
}

export interface ClusterInfo {
  id: number
  name: string
  size: number
  keywords: string[]
  documents: any[]
}

export interface InsightInfo {
  insights: string[]
  summary: string
  cluster_count: number
  total_documents: number
}

export interface ResearchResult {
  request_id: string
  query: string
  status: string
  clusters: ClusterInfo[]
  insights?: InsightInfo
  created_at: string
  completed_at?: string
}

export type ResearchStatus =
  | 'idle'
  | 'searching'
  | 'analyzing'
  | 'clustering'
  | 'generating_insights'
  | 'completed'
  | 'failed'
```

**Step 2: Create API client**

Create `frontend/src/api/research.ts`:

```typescript
import { ResearchRequest, ResearchResponse, ResearchResult } from '../types'

const API_BASE = '/api'

export async function createResearchRequest(
  query: string
): Promise<ResearchResponse> {
  const response = await fetch(`${API_BASE}/research`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  })

  if (!response.ok) {
    throw new Error('Failed to create research request')
  }

  return response.json()
}

export async function getResearchResult(
  requestId: string
): Promise<ResearchResult> {
  const response = await fetch(`${API_BASE}/research/${requestId}`)

  if (!response.ok) {
    throw new Error('Failed to fetch research result')
  }

  return response.json()
}

export function streamResearchProgress(
  requestId: string,
  onProgress: (status: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): EventSource {
  const eventSource = new EventSource(
    `${API_BASE}/research/${requestId}/stream`
  )

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onProgress(data.status)

      if (data.status === 'completed' || data.status === 'failed') {
        eventSource.close()
        onComplete()
      }
    } catch (error) {
      onError(error as Error)
      eventSource.close()
    }
  }

  eventSource.onerror = (error) => {
    onError(new Error('SSE connection error'))
    eventSource.close()
  }

  return eventSource
}
```

**Step 3: Update App.tsx with API integration**

Modify `frontend/src/App.tsx`:

```typescript
import React, { useState } from 'react'
import { createResearchRequest, streamResearchProgress, getResearchResult } from './api/research'
import { ResearchResult, ResearchStatus } from './types'

function App() {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<ResearchStatus>('idle')
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    setStatus('searching')
    setError(null)
    setResult(null)

    try {
      // Create research request
      const response = await createResearchRequest(query)

      // Stream progress updates
      streamResearchProgress(
        response.request_id,
        (newStatus) => {
          setStatus(newStatus as ResearchStatus)
        },
        async () => {
          // Fetch final results
          const finalResult = await getResearchResult(response.request_id)
          setResult(finalResult)
          setStatus('completed')
        },
        (err) => {
          setError(err.message)
          setStatus('failed')
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setStatus('failed')
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Market Research Platform</h1>
      </header>

      <main>
        <div className="search-box">
          <input
            type="text"
            placeholder="Enter topic to research..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={status !== 'idle' && status !== 'completed' && status !== 'failed'}
          />
          <button
            onClick={handleSearch}
            disabled={status !== 'idle' && status !== 'completed' && status !== 'failed'}
          >
            Research
          </button>
        </div>

        {status !== 'idle' && status !== 'completed' && (
          <div className="status">
            <p>Status: {status}</p>
          </div>
        )}

        {error && (
          <div className="error">
            <p>Error: {error}</p>
          </div>
        )}

        {result && (
          <div className="results">
            <h2>Results for: {result.query}</h2>
            <p>Found {result.clusters.length} topics</p>

            {result.insights && (
              <div className="insights">
                <h3>Key Insights</h3>
                <ul>
                  {result.insights.insights.map((insight, i) => (
                    <li key={i}>{insight}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="clusters">
              {result.clusters.map((cluster) => (
                <div key={cluster.id} className="cluster-card">
                  <h3>{cluster.name}</h3>
                  <p>{cluster.size} documents</p>
                  <div className="keywords">
                    {cluster.keywords.map((kw, i) => (
                      <span key={i} className="keyword">{kw}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
```

**Step 4: Update CSS for new components**

Add to `frontend/src/index.css`:

```css
.status {
  max-width: 600px;
  margin: 2rem auto;
  padding: 1rem;
  background: #e7f3ff;
  border-radius: 8px;
  text-align: center;
}

.error {
  max-width: 600px;
  margin: 2rem auto;
  padding: 1rem;
  background: #ffe7e7;
  border-radius: 8px;
  text-align: center;
  color: #d00;
}

.results {
  max-width: 1200px;
  margin: 2rem auto;
}

.insights {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.insights h3 {
  margin-bottom: 1rem;
}

.insights ul {
  list-style-position: inside;
}

.insights li {
  margin: 0.5rem 0;
}

.clusters {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.cluster-card {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.cluster-card h3 {
  margin-bottom: 0.5rem;
  color: #333;
}

.cluster-card p {
  color: #666;
  margin-bottom: 1rem;
}

.keywords {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.keyword {
  background: #e7f3ff;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  color: #0066cc;
}
```

**Step 5: Test the integration**

Note: This requires backend to be running. Manual test:
1. Start backend: `cd backend && poetry run uvicorn src.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to http://localhost:3000
4. Try searching for a topic

**Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: implement API integration and SSE streaming

Connected frontend to backend with real-time progress updates"
```

---

## Phase 5: Visualization Components

### Task 11: Cluster Visualization with Recharts

**Files:**
- Create: `frontend/src/components/ClusterMap.tsx`
- Create: `frontend/src/components/TrendTimeline.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: Create ClusterMap component**

Create `frontend/src/components/ClusterMap.tsx`:

```typescript
import React from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts'
import { ClusterInfo } from '../types'

interface ClusterMapProps {
  clusters: ClusterInfo[]
}

const COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1',
  '#d084d0', '#82ca82', '#ffc042', '#ff5042', '#8db1e1'
]

export function ClusterMap({ clusters }: ClusterMapProps) {
  // Transform clusters to scatter plot data
  const data = clusters.map((cluster, index) => ({
    x: index * 100 + Math.random() * 50,
    y: Math.random() * 100,
    size: cluster.size * 20,
    name: cluster.name,
    cluster: cluster
  }))

  return (
    <div className="cluster-map">
      <h3>Cluster Visualization</h3>
      <ScatterChart
        width={800}
        height={400}
        margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
      >
        <CartesianGrid />
        <XAxis type="number" dataKey="x" name="x" hide />
        <YAxis type="number" dataKey="y" name="y" hide />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null
            const data = payload[0].payload
            return (
              <div className="custom-tooltip">
                <p><strong>{data.name}</strong></p>
                <p>{data.cluster.size} documents</p>
              </div>
            )
          }}
        />
        <Scatter data={data} fill="#8884d8">
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Scatter>
      </ScatterChart>
    </div>
  )
}
```

**Step 2: Create TrendTimeline component**

Create `frontend/src/components/TrendTimeline.tsx`:

```typescript
import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'
import { ClusterInfo } from '../types'

interface TrendTimelineProps {
  clusters: ClusterInfo[]
}

export function TrendTimeline({ clusters }: TrendTimelineProps) {
  // Mock timeline data (in real implementation, extract from published_date)
  const data = Array.from({ length: 7 }, (_, i) => {
    const entry: any = { date: `Day ${i + 1}` }
    clusters.forEach((cluster) => {
      entry[cluster.name] = Math.floor(Math.random() * cluster.size)
    })
    return entry
  })

  const colors = [
    '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1'
  ]

  return (
    <div className="trend-timeline">
      <h3>Trend Timeline</h3>
      <LineChart
        width={800}
        height={300}
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        {clusters.map((cluster, index) => (
          <Line
            key={cluster.id}
            type="monotone"
            dataKey={cluster.name}
            stroke={colors[index % colors.length]}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </div>
  )
}
```

**Step 3: Update App.tsx to use visualization components**

Modify `frontend/src/App.tsx` to import and use the components:

```typescript
import React, { useState } from 'react'
import { createResearchRequest, streamResearchProgress, getResearchResult } from './api/research'
import { ResearchResult, ResearchStatus } from './types'
import { ClusterMap } from './components/ClusterMap'
import { TrendTimeline } from './components/TrendTimeline'

function App() {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<ResearchStatus>('idle')
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    setStatus('searching')
    setError(null)
    setResult(null)

    try {
      const response = await createResearchRequest(query)

      streamResearchProgress(
        response.request_id,
        (newStatus) => {
          setStatus(newStatus as ResearchStatus)
        },
        async () => {
          const finalResult = await getResearchResult(response.request_id)
          setResult(finalResult)
          setStatus('completed')
        },
        (err) => {
          setError(err.message)
          setStatus('failed')
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setStatus('failed')
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Market Research Platform</h1>
      </header>

      <main>
        <div className="search-box">
          <input
            type="text"
            placeholder="Enter topic to research..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={status !== 'idle' && status !== 'completed' && status !== 'failed'}
          />
          <button
            onClick={handleSearch}
            disabled={status !== 'idle' && status !== 'completed' && status !== 'failed'}
          >
            Research
          </button>
        </div>

        {status !== 'idle' && status !== 'completed' && (
          <div className="status">
            <p>Status: {status}</p>
          </div>
        )}

        {error && (
          <div className="error">
            <p>Error: {error}</p>
          </div>
        )}

        {result && (
          <div className="results">
            <h2>Results for: {result.query}</h2>

            {result.insights && (
              <div className="insights">
                <h3>Key Insights</h3>
                <ul>
                  {result.insights.insights.map((insight, i) => (
                    <li key={i}>{insight}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="visualizations">
              <ClusterMap clusters={result.clusters} />
              <TrendTimeline clusters={result.clusters} />
            </div>

            <div className="clusters">
              {result.clusters.map((cluster) => (
                <div key={cluster.id} className="cluster-card">
                  <h3>{cluster.name}</h3>
                  <p>{cluster.size} documents</p>
                  <div className="keywords">
                    {cluster.keywords.map((kw, i) => (
                      <span key={i} className="keyword">{kw}</span>
                    ))}
                  </div>
                  <div className="documents">
                    {cluster.documents.slice(0, 3).map((doc, i) => (
                      <div key={i} className="document">
                        <a href={doc.url} target="_blank" rel="noopener noreferrer">
                          {doc.title}
                        </a>
                        <span className="source">{doc.source}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
```

**Step 4: Add visualization styles**

Add to `frontend/src/index.css`:

```css
.visualizations {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.cluster-map,
.trend-timeline {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.cluster-map h3,
.trend-timeline h3 {
  margin-bottom: 1rem;
}

.custom-tooltip {
  background: white;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.documents {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #eee;
}

.document {
  margin: 0.5rem 0;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.document a {
  color: #0066cc;
  text-decoration: none;
  font-size: 0.875rem;
}

.document a:hover {
  text-decoration: underline;
}

.source {
  font-size: 0.75rem;
  color: #999;
}
```

**Step 5: Test visualization components**

Manual test with backend running

**Step 6: Commit**

```bash
git add frontend/src/components/ frontend/src/App.tsx frontend/src/index.css
git commit -m "feat: add cluster map and trend timeline visualizations

Implemented interactive scatter plot and timeline charts with Recharts"
```

---

## Phase 6: Testing & Documentation

### Task 12: Add Comprehensive Tests

**Files:**
- Create: `backend/tests/test_integration.py`
- Create: `frontend/src/App.test.tsx`

**Step 1: Create integration tests**

Create `backend/tests/test_integration.py`:

```python
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
```

**Step 2: Run integration tests**

Run: `cd backend && poetry run pytest tests/test_integration.py -v -m integration`
Expected: PASS

**Step 3: Create README**

Create `README.md`:

```markdown
# Market Research Platform

AI-powered market research platform that analyzes trends using Tavily search, LangGraph workflows, and automatic topic clustering.

## Features

- Real-time trend analysis via Tavily API
- Automatic sub-topic discovery using HDBSCAN clustering
- LLM-powered insights generation
- Interactive visualizations (cluster maps, trend timelines)
- SSE streaming for real-time progress updates
- 24-hour result caching for performance

## Tech Stack

**Backend:**
- FastAPI
- LangGraph (multi-agent orchestration)
- SQLite + SQLAlchemy
- Tavily API (search)
- sentence-transformers (embeddings)
- scikit-learn (clustering)

**Frontend:**
- React + TypeScript
- Vite
- Recharts (visualizations)

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key
- Tavily API key

### Backend Setup

```bash
cd backend
poetry install
cp .env.example .env
# Edit .env with your API keys
poetry run uvicorn src.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Navigate to http://localhost:3000
2. Enter a topic to research (e.g., "AI agents")
3. Watch real-time progress as the system:
   - Searches for latest information
   - Analyzes and clusters results
   - Generates insights
4. Explore interactive visualizations and cluster details

## API Documentation

- `POST /api/research` - Create new research request
- `GET /api/research/{id}` - Get research results
- `GET /api/research/{id}/stream` - Stream progress via SSE
- `GET /health` - Health check

## Testing

```bash
# Backend tests
cd backend
poetry run pytest

# Frontend tests
cd frontend
npm test
```

## Architecture

See `docs/plans/2025-11-19-market-research-platform-design.md` for detailed design documentation.

## License

MIT
```

**Step 4: Commit**

```bash
git add backend/tests/test_integration.py README.md
git commit -m "docs: add integration tests and README

Comprehensive documentation for setup and usage"
```

---

## Execution Options

Plan complete and saved to `docs/plans/2025-11-19-market-research-platform-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
