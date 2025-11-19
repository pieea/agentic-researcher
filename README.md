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

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key
- Tavily API key

### One-Command Startup

```bash
# 1. Set up environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# 2. Start everything (Unix/Mac)
./start.sh

# Or on Windows
start.bat
```

The startup script will automatically:
- Install backend dependencies (if needed)
- Install frontend dependencies (if needed)
- Start backend server on port 8000
- Start frontend server on port 3000
- Display all service URLs

To stop all servers:
```bash
./stop.sh
```

### Manual Setup (Alternative)

<details>
<summary>Click to expand manual setup instructions</summary>

#### Backend Setup

```bash
cd backend
uv sync
cp .env.example .env
# Edit .env with your API keys
uv run uvicorn src.main:app --reload
```

#### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

</details>

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
uv run pytest

# Frontend tests
cd frontend
npm test
```

## Architecture

See `docs/plans/2025-11-19-market-research-platform-design.md` for detailed design documentation.

## License

MIT
