from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class InsightAgent:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7
        )
        logger.info(f"Initialized InsightAgent with model: {model}")

    def generate_insights(
        self,
        query: str,
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate key insights from clustered research results.

        Args:
            query: Original search query
            clusters: List of cluster information (name, size, keywords)

        Returns:
            Dictionary with insights, trends, and summary
        """
        logger.info(f"Generating insights for query: {query}")

        # Prepare cluster summary
        cluster_summary = "\n".join([
            f"- {c['name']}: {c['size']} documents, keywords: {', '.join(c['keywords'][:5])}"
            for c in clusters
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert market research analyst.
            Analyze the following research clusters and provide 3-5 key insights about the topic.
            Focus on trends, patterns, and actionable takeaways.
            Be concise and specific."""),
            ("user", """Query: {query}

Clusters found:
{clusters}

Provide 3-5 key insights as a numbered list.""")
        ])

        try:
            response = self.llm.invoke(
                prompt.format_messages(query=query, clusters=cluster_summary)
            )

            # Parse insights from response
            insights_text = response.content
            insights = [
                line.strip()
                for line in insights_text.split('\n')
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))
            ]

            return {
                "insights": insights,
                "summary": insights_text,
                "cluster_count": len(clusters),
                "total_documents": sum(c['size'] for c in clusters)
            }

        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            # Fallback to basic statistics
            return {
                "insights": [
                    f"Found {len(clusters)} major topics related to '{query}'",
                    f"Analyzed {sum(c['size'] for c in clusters)} documents total",
                    f"Largest topic: {max(clusters, key=lambda x: x['size'])['name']}"
                ],
                "summary": "Basic statistical summary (LLM unavailable)",
                "cluster_count": len(clusters),
                "total_documents": sum(c['size'] for c in clusters)
            }

    def generate_cluster_names(
        self,
        keywords_list: List[List[str]]
    ) -> List[str]:
        """
        Generate meaningful names for clusters based on keywords.

        Args:
            keywords_list: List of keyword lists, one per cluster

        Returns:
            List of generated cluster names
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a naming expert. Create concise, descriptive names (2-4 words) for topics based on keywords."),
            ("user", "Keywords: {keywords}\n\nGenerate a concise topic name:")
        ])

        names = []
        for keywords in keywords_list:
            try:
                response = self.llm.invoke(
                    prompt.format_messages(keywords=", ".join(keywords))
                )
                name = response.content.strip().strip('"').strip("'")
                names.append(name)
            except:
                # Fallback to simple name
                names.append(f"Topic: {keywords[0] if keywords else 'Unknown'}")

        return names
