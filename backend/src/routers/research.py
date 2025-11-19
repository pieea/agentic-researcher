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
    """Execute the research workflow with real-time progress updates."""
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
        logger.info(f"[{request_id}] Initialized workflow, stored in active_workflows")

        # Execute workflow with streaming to capture intermediate states
        final_state = None
        for output in workflow.stream(initial_state):
            # LangGraph stream returns dict with node name as key
            # Extract the state from the output
            for node_name, node_state in output.items():
                current_status = node_state.get('status', 'unknown')
                logger.info(f"[{request_id}] Node '{node_name}' completed with status: {current_status}")

                # Update active workflow state in real-time
                active_workflows[request_id] = node_state
                logger.info(f"[{request_id}] Updated active_workflows with status: {current_status}")
                final_state = node_state

                # Delay to ensure SSE can catch the update (longer than SSE polling interval)
                await asyncio.sleep(0.5)

        # Update database with final status
        if final_state:
            db_query = db.query(DBResearchQuery).filter_by(id=int(request_id)).first()
            if db_query:
                db_query.status = final_state.get("status", "completed")
                db.commit()
            logger.info(f"[{request_id}] Workflow completed with final status: {final_state.get('status')}")

    except Exception as e:
        logger.error(f"Workflow execution failed for {request_id}: {str(e)}")

        # Update in-memory state
        if request_id in active_workflows:
            active_workflows[request_id]["status"] = "failed"
            active_workflows[request_id]["error"] = str(e)

        # Update database
        db_query = db.query(DBResearchQuery).filter_by(id=int(request_id)).first()
        if db_query:
            db_query.status = "failed"
            db.commit()

@router.get("/{request_id}/stream")
async def stream_progress(request_id: str):
    """Stream progress updates via SSE with detailed node information."""

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for progress updates."""

        last_status = None

        while True:
            if request_id not in active_workflows:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            state = active_workflows[request_id]
            status = state.get("status", "unknown")

            # Only send update if status changed or has useful information
            if status != last_status:
                progress_data = {
                    "status": status,
                    "query": state.get("query"),
                }

                # Add node-specific details based on current status
                if status == "initialized":
                    progress_data["message"] = "분석 준비 중..."
                    progress_data["node"] = "search"

                elif status == "searching":
                    progress_data["message"] = "검색 중..."
                    progress_data["node"] = "search"

                elif status == "search_completed":
                    results_count = len(state.get("raw_results", []))
                    progress_data["message"] = f"검색 완료 ({results_count}개 결과)"
                    progress_data["node"] = "search"
                    progress_data["results_count"] = results_count

                elif status == "analyzing":
                    progress_data["message"] = "데이터 분석 중..."
                    progress_data["node"] = "analysis"

                elif status in ["clustering_completed", "clustering_skipped"]:
                    clusters_count = len(state.get("clusters", []))
                    progress_data["message"] = f"클러스터링 완료 ({clusters_count}개 주제)"
                    progress_data["node"] = "analysis"
                    progress_data["clusters_count"] = clusters_count

                elif status == "generating_insights":
                    progress_data["message"] = "인사이트 생성 중..."
                    progress_data["node"] = "insight"

                elif status == "completed":
                    progress_data["message"] = "분석 완료"
                    progress_data["node"] = "insight"
                    progress_data["clusters_count"] = len(state.get("clusters", []))
                    progress_data["insights_count"] = len(state.get("insights", {}).get("insights", []))

                elif status == "failed":
                    progress_data["message"] = "오류 발생"
                    progress_data["error"] = state.get("error", "Unknown error")

                yield f"data: {json.dumps(progress_data)}\n\n"
                last_status = status

            if status in ["completed", "failed"]:
                break

            await asyncio.sleep(0.1)  # Poll every 100ms for real-time updates

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
