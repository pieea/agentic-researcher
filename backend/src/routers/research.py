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
    db_query = DBResearchQuery(
        query=request.query,
        status="processing"
    )
    db.add(db_query)
    db.commit()
    db.refresh(db_query)

    request_id = str(db_query.id)

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
        db_query = db.query(DBResearchQuery).filter_by(id=int(request_id)).first()
        if db_query:
            db_query.status = final_state.get("status", "completed")
            db.commit()

        # Store final state
        active_workflows[request_id] = final_state

    except Exception as e:
        logger.error(f"Workflow execution failed for {request_id}: {str(e)}")

        # Update database
        db_query = db.query(DBResearchQuery).filter_by(id=int(request_id)).first()
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

    db_query = db.query(DBResearchQuery).filter_by(id=int(request_id)).first()

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
