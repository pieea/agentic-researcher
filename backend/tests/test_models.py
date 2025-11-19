import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database import Base, get_db
from src.models import ResearchQuery, SearchResult, Cluster, ClusterMembership

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

def test_create_research_query(test_db):
    query = ResearchQuery(query="AI agent", status="processing")
    test_db.add(query)
    test_db.commit()

    result = test_db.query(ResearchQuery).filter_by(query="AI agent").first()
    assert result is not None
    assert result.query == "AI agent"
    assert result.status == "processing"
