from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta, timezone
from src.database import Base

class ResearchQuery(Base):
    __tablename__ = "research_queries"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime, default=lambda: datetime.now(timezone.utc) + timedelta(hours=24), nullable=False)
    status = Column(String, default="processing", nullable=False)  # processing, completed, failed

    search_results = relationship("SearchResult", back_populates="query")
    clusters = relationship("Cluster", back_populates="query")

class SearchResult(Base):
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("research_queries.id"), nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    summary = Column(Text, nullable=False)
    published_date = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    relevance_score = Column(Float, default=1.0, nullable=False)

    query = relationship("ResearchQuery", back_populates="search_results")
    cluster_memberships = relationship("ClusterMembership", back_populates="result")

class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("research_queries.id"), nullable=False)
    cluster_name = Column(String, nullable=False)
    size = Column(Integer, default=0, nullable=False)
    keywords = Column(JSON, nullable=False)

    query = relationship("ResearchQuery", back_populates="clusters")
    memberships = relationship("ClusterMembership", back_populates="cluster")

class ClusterMembership(Base):
    __tablename__ = "cluster_memberships"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    result_id = Column(Integer, ForeignKey("search_results.id"), nullable=False)
    distance = Column(Float, nullable=False)

    cluster = relationship("Cluster", back_populates="memberships")
    result = relationship("SearchResult", back_populates="cluster_memberships")
