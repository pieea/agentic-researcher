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
