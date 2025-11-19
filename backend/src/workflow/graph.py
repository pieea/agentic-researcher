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

            # Check if we got any results
            if not results or len(results) == 0:
                logger.warning(f"No search results found for query: '{state['query']}'")
                state["raw_results"] = []
                state["status"] = "failed"
                state["error"] = "검색 결과를 찾을 수 없습니다. 다른 키워드로 다시 시도해주세요."
                return state

            state["raw_results"] = results
            state["status"] = "search_completed"
            logger.info(f"Found {len(results)} results")
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
            # Check if we have results to analyze
            if not state.get("raw_results") or len(state["raw_results"]) == 0:
                logger.error("No search results to analyze")
                state["status"] = "failed"
                state["error"] = "분석할 검색 결과가 없습니다."
                return state

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
                labels = analysis_agent.cluster_embeddings(embeddings, min_cluster_size=2)
                state["cluster_labels"] = labels

                # Get unique cluster IDs (K-means doesn't have noise, HDBSCAN uses -1 for noise)
                unique_clusters = sorted([label for label in set(labels) if label != -1])

                # Check if we have any valid clusters
                if not unique_clusters:
                    # All points are noise, treat as single cluster
                    logger.warning("All documents classified as noise, creating single cluster")
                    state["clusters"] = [{
                        "id": 0,
                        "name": state["query"],
                        "size": len(state["raw_results"]),
                        "keywords": [],
                        "documents": state["raw_results"][:3]
                    }]
                    state["status"] = "clustering_skipped"
                else:
                    # Extract keywords per cluster
                    keywords_map = analysis_agent.extract_cluster_keywords(texts, labels, top_k=5)

                    # Generate cluster names only for existing clusters
                    keywords_list = [keywords_map.get(cluster_id, []) for cluster_id in unique_clusters]
                    cluster_names = insight_agent.generate_cluster_names(keywords_list)

                    # Build cluster info
                    clusters = []
                    for idx, cluster_id in enumerate(unique_clusters):
                        cluster_docs = [
                            state["raw_results"][i]
                            for i, label in enumerate(labels)
                            if label == cluster_id
                        ]

                        # Only add clusters that have documents
                        if cluster_docs:
                            clusters.append({
                                "id": int(cluster_id),  # Convert numpy.int32 to Python int
                                "name": cluster_names[idx] if idx < len(cluster_names) else f"주제 {cluster_id}",
                                "size": len(cluster_docs),
                                "keywords": keywords_map.get(cluster_id, []),
                                "documents": cluster_docs[:3]  # Top 3 representative docs
                            })

                    state["clusters"] = clusters
                    state["status"] = "clustering_completed"
                    logger.info(f"Created {len(clusters)} clusters from {len(unique_clusters)} unique labels")
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

    # Define conditional routing function
    def should_continue_after_search(state: ResearchState) -> str:
        """Decide whether to continue to analysis or end."""
        if state.get("status") == "failed":
            logger.info("Search failed, skipping analysis and insight generation")
            return "end"
        return "continue"

    # Build graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("insight", insight_node)

    # Define edges with conditional routing
    workflow.set_entry_point("search")
    workflow.add_conditional_edges(
        "search",
        should_continue_after_search,
        {
            "continue": "analysis",
            "end": END
        }
    )
    workflow.add_edge("analysis", "insight")
    workflow.add_edge("insight", END)

    return workflow.compile()
