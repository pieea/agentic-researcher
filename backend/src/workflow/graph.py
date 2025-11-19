from langgraph.graph import StateGraph, END
from src.workflow.state import ResearchState
from src.agents.search_agent import SearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.insight_agent import InsightAgent
from src.config import Settings
import logging

logger = logging.getLogger(__name__)

def create_research_workflow() -> StateGraph:
    """Create the LangGraph workflow for market research."""

    # Initialize settings
    settings = Settings()

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
