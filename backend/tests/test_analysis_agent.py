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
