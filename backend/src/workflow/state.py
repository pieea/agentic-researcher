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
