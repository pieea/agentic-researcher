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
            Dictionary with insights, success cases, failure cases, and market outlook
        """
        logger.info(f"Generating insights for query: {query}")

        # Prepare cluster summary
        cluster_summary = "\n".join([
            f"- {c['name']}: {c['size']} documents, keywords: {', '.join(c['keywords'][:5])}"
            for c in clusters
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert market research analyst.
            Analyze the following research clusters and provide comprehensive market insights.
            Be specific, data-driven, and actionable."""),
            ("user", """Query: {query}

Clusters found:
{clusters}

Please provide a comprehensive analysis in the following format:

## 핵심 인사이트
(3-5 key insights as numbered list)

## 성공 사례
(2-3 success stories or best practices)

## 실패 사례
(2-3 failure cases or lessons learned)

## 향후 시장 전망
(Market outlook and future trends in 2-3 points)

Use Korean for all sections. Be specific and concise.""")
        ])

        try:
            response = self.llm.invoke(
                prompt.format_messages(query=query, clusters=cluster_summary)
            )

            # Parse response into sections
            content = response.content
            sections = {
                "insights": [],
                "success_cases": [],
                "failure_cases": [],
                "market_outlook": []
            }

            current_section = None
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Detect section headers
                if '핵심 인사이트' in line or 'Key Insights' in line:
                    current_section = 'insights'
                elif '성공 사례' in line or '성공사례' in line or 'Success' in line:
                    current_section = 'success_cases'
                elif '실패 사례' in line or '실패사례' in line or 'Failure' in line:
                    current_section = 'failure_cases'
                elif '향후 시장 전망' in line or '시장 전망' in line or 'Market Outlook' in line:
                    current_section = 'market_outlook'
                # Add content to current section
                elif current_section and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and bullets
                    clean_line = line.lstrip('0123456789.-•*) ').strip()
                    if clean_line:
                        sections[current_section].append(clean_line)

            # Ensure we have at least some content
            if not sections["insights"]:
                sections["insights"] = [
                    f"'{query}' 관련 {len(clusters)}개의 주요 주제 발견",
                    f"총 {sum(c['size'] for c in clusters)}개 문서 분석",
                    f"가장 큰 주제: {max(clusters, key=lambda x: x['size'])['name']}"
                ]

            return {
                "insights": sections["insights"],
                "success_cases": sections["success_cases"],
                "failure_cases": sections["failure_cases"],
                "market_outlook": sections["market_outlook"],
                "summary": content,
                "cluster_count": len(clusters),
                "total_documents": sum(c['size'] for c in clusters)
            }

        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            # Fallback to basic statistics
            return {
                "insights": [
                    f"'{query}' 관련 {len(clusters)}개의 주요 주제 발견",
                    f"총 {sum(c['size'] for c in clusters)}개 문서 분석",
                    f"가장 큰 주제: {max(clusters, key=lambda x: x['size'])['name']}"
                ],
                "success_cases": ["상세 분석을 위해서는 LLM이 필요합니다"],
                "failure_cases": ["상세 분석을 위해서는 LLM이 필요합니다"],
                "market_outlook": ["상세 분석을 위해서는 LLM이 필요합니다"],
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
