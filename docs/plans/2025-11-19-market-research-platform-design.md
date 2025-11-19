# Market Research Platform Design

**Date:** 2025-11-19
**Status:** Approved

## 1. System Overview

### Purpose
AI-powered market research platform that provides real-time trend analysis and insights. Users input topics of interest, and the system automatically collects latest information, discovers sub-topics through clustering, and presents visual insights.

### Architecture
```
[React Frontend] ←→ [FastAPI Backend] ←→ [LangGraph Workflow]
                           ↓
                      [SQLite DB]
                           ↓
                    [Tavily Search API]
```

### Technology Stack
- **Frontend:** React + TypeScript + Recharts/D3.js (visualization)
- **Backend:** FastAPI + Python 3.11+
- **Orchestration:** LangGraph (multi-agent workflow)
- **Database:** SQLite + SQLAlchemy ORM
- **Search:** Tavily API
- **Embeddings:** OpenAI embeddings or sentence-transformers
- **Clustering:** scikit-learn (KMeans or HDBSCAN)

### Execution Flow
1. User inputs search query (e.g., "AI agent")
2. FastAPI starts LangGraph workflow
3. Search agent collects latest materials via Tavily
4. Analysis agent generates embeddings + clustering
5. Insight agent produces structured results
6. React displays streaming progress and final results
7. Results cached in SQLite (fast response on re-request)

---

## 2. LangGraph Multi-Agent Workflow

### Agent Composition

#### 1. Search Agent
- **Role:** Collect latest information via Tavily API
- **Input:** Search query, result count (default 20-30)
- **Output:** Raw search results list (title, URL, summary, published date)
- **Logic:** Apply recency weighting (higher scores for last 7 days)

#### 2. Analysis Agent
- **Role:** Generate text embeddings and clustering
- **Input:** Search results list
- **Output:** Clustered groups (auto-generated sub-topic names per group)
- **Logic:**
  - Convert title + summary text to embeddings
  - Discover cluster count automatically with HDBSCAN
  - Extract representative keywords per cluster
  - Generate meaningful topic names per cluster using LLM

#### 3. Insight Agent
- **Role:** Trend analysis and key insights extraction
- **Input:** Clustering results
- **Output:** Major trends, temporal changes, key points
- **Logic:**
  - Analyze mention volume over time
  - Compare cluster sizes (identify major sub-topics)
  - Generate 3-5 key insight summaries using LLM

### LangGraph Node Structure
```
START → SearchNode → AnalysisNode → InsightNode → VisualizationNode → END
          ↓             ↓              ↓               ↓
        (update)      (update)       (update)    (final result)
```

### State Schema
```python
class ResearchState(TypedDict):
    query: str
    raw_results: List[SearchResult]
    embeddings: List[Vector]
    clusters: List[Cluster]
    insights: InsightSummary
    visualization_data: Dict
    status: str  # "searching", "analyzing", "clustering", "generating_insights"
```

---

## 3. Data Model & API Design

### SQLite Database Schema

```python
# Research request and result caching
class ResearchQuery(Base):
    id: int (PK)
    query: str (indexed)
    created_at: datetime
    expires_at: datetime  # created_at + 24 hours
    status: str  # "processing", "completed", "failed"

# Retrieved source materials
class SearchResult(Base):
    id: int (PK)
    query_id: int (FK)
    title: str
    url: str (unique)
    summary: str
    published_date: datetime
    source: str
    relevance_score: float

# Cluster information
class Cluster(Base):
    id: int (PK)
    query_id: int (FK)
    cluster_name: str  # LLM-generated sub-topic name
    size: int  # number of documents
    keywords: JSON  # representative keyword list

# Cluster-document mapping
class ClusterMembership(Base):
    cluster_id: int (FK)
    result_id: int (FK)
    distance: float  # distance from cluster center
```

### FastAPI Endpoints

```
POST /api/research
- Body: { "query": "AI agent" }
- Response: { "request_id": "uuid", "status": "processing" }

GET /api/research/{request_id}/stream
- SSE streaming for progress updates
- Events: "searching", "analyzing", "clustering", "completed"

GET /api/research/{request_id}
- Return final results
- Response: {
    "query": "AI agent",
    "clusters": [...],
    "insights": {...},
    "visualization": {...},
    "sources": [...]
  }

GET /api/research/cache/{query_hash}
- Check cached results (within 24 hours)
```

---

## 4. Visualization Design

### Visualization Components

#### 1. Cluster Map
- **Library:** D3.js force-directed graph or Recharts scatter plot
- **Representation:** Each cluster as node, size proportional to document count
- **Interaction:** Click to display document list for that cluster
- **Color:** Auto-assigned per cluster

#### 2. Trend Timeline
- **Library:** Recharts line chart
- **X-axis:** Time (recent 7-30 days)
- **Y-axis:** Mention volume
- **Lines:** Color-coded by cluster
- **Recency emphasis:** Highlight last 48 hours

#### 3. Cluster Detail Cards
- **Layout:** Grid format
- **Card contents:**
  - Sub-topic name (LLM-generated)
  - Document count and percentage
  - Key keywords (tag format)
  - Top 3 representative documents (title, source, date)

#### 4. Key Insights Panel
- **Layout:** Top fixed panel
- **Content:** 3-5 bullet points
- **Example:** "Autonomous agents increased 48% in AI agent market"

### Source Citation Method
- Display source URL + published date on all document cards
- Click to navigate to original link
- Show source-specific icons (news/blog/research)

---

## 5. Error Handling & Testing

### Error Handling Strategy

#### 1. Tavily API Failure
- Retry logic: 3 retries with exponential backoff
- Fallback: Suggest cached similar query results
- User message: "Search service temporary error, retrying..."

#### 2. Clustering Failure
- Cause: Too few results (< 5)
- Fallback: Display in chronological order without clustering
- User message: "Skipping clustering due to limited results"

#### 3. LLM API Timeout/Failure
- Insight generation failure: Display basic statistics only
- Cluster name generation failure: Use default names "Topic 1", "Topic 2", etc.

#### 4. Streaming Connection Lost
- Implement reconnection logic
- Resume from last state

### Testing Strategy

#### 1. Unit Tests
- Test each LangGraph node independently
- Mock Tavily API responses
- Test clustering algorithm accuracy

#### 2. Integration Tests
- End-to-end full workflow testing
- Actual Tavily API calls (using test credits)
- DB transaction validation

#### 3. Frontend Tests
- React component rendering tests (Jest + RTL)
- Visualization snapshot tests
- SSE streaming connection tests

### Performance Goals
- Search → Final results: Within 30 seconds
- Cache hit: Within 1 second
- Concurrent request handling: Minimum 10

---

## 6. Implementation Priorities

### Phase 1: Core Backend (MVP)
1. FastAPI setup + basic endpoints
2. LangGraph workflow (Search → Analysis → Insight nodes)
3. Tavily API integration
4. SQLite schema + basic caching

### Phase 2: Clustering & Analysis
1. Embedding generation (sentence-transformers)
2. HDBSCAN clustering implementation
3. LLM-based cluster naming
4. Insight generation logic

### Phase 3: Frontend
1. React app setup + TypeScript configuration
2. Basic UI (search input + results display)
3. SSE streaming integration
4. Visualization components (Recharts/D3.js)

### Phase 4: Polish & Optimization
1. Error handling + retry logic
2. Caching optimization
3. Performance tuning
4. Testing suite

---

## Design Decisions & Rationale

### Why Real-time Analysis?
Recency is a core requirement, making real-time analysis essential. Caching optimizes performance and cost while maintaining freshness.

### Why Multi-Agent Architecture?
Separation of concerns improves maintainability and extensibility. Each agent can be tested, optimized, and replaced independently.

### Why SQLite?
Simple setup, zero configuration, easy deployment. Sufficient for MVP and medium-scale usage. Can migrate to PostgreSQL later if needed.

### Why HDBSCAN?
Automatically determines cluster count, handles noise well, and works effectively with varying density clusters.

### Why Tavily?
Optimized for AI agents, provides clean structured results, handles recency weighting natively.
