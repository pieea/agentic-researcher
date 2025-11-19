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
