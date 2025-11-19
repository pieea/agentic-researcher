from tavily import TavilyClient
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SearchAgent:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        """
        Execute search via Tavily API with recency weighting.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, content, score, published_date
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_domains=[],
                exclude_domains=[]
            )

            results = []
            for item in response.get("results", []):
                # Apply recency weighting
                published_date = item.get("published_date")
                recency_boost = self._calculate_recency_boost(published_date)

                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0) * recency_boost,
                    "published_date": published_date,
                    "source": self._extract_domain(item.get("url", ""))
                })

            # Sort by adjusted score
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Search completed: {len(results)} results for query '{query}'")
            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            raise

    def _calculate_recency_boost(self, published_date: str) -> float:
        """Apply higher weight to recent content (last 7 days)."""
        if not published_date:
            return 1.0

        try:
            pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            days_ago = (datetime.now(pub_date.tzinfo) - pub_date).days

            if days_ago <= 2:
                return 1.5  # Last 2 days: 50% boost
            elif days_ago <= 7:
                return 1.2  # Last week: 20% boost
            else:
                return 1.0  # Older: no boost
        except:
            return 1.0

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
