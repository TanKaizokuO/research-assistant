import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import app
from output_schemas.schema import SourceSchema

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns 200 OK and status 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("services.research_service.LLM")
@patch("services.research_service.manual_web_search")
@patch("services.research_service.fetch_and_save_best_arxiv_paper")
@patch("services.research_service.paper_search")
def test_research_endpoint(mock_ss_search, mock_arxiv_search, mock_web_search, mock_llm):
    """Test the /research/ endpoint with mocked external service calls."""
    mock_ss_search.return_value = [
        {
            "title": "Mocked Paper 1",
            "abstract": "This is a mocked abstract about research.",
            "year": 2026,
            "citations": 10,
            "pdf": "http://example.com/paper.pdf"
        }
    ]
    
    mock_arxiv_search.return_value = {
        "saved_papers": [
            {
                "title": "Mocked arXiv Paper",
                "file_path": "/mock/path/arxiv.md"
            }
        ]
    }
    
    mock_web_search.return_value = [
        SourceSchema(
            source_id="mock_web_page",
            url="http://example.com/web",
            title="Mock Web Page",
            source_type="web_article",
            retrieval_timestamp=datetime.now(timezone.utc),
            summary="A web page about research topics and artificial intelligence.",
            confidence_score=95
        )
    ]
    
    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is a generated research brief based on mocked data."
    mock_llm.invoke.return_value = mock_llm_response

    response = client.post(
        "/research/",
        json={
            "topic": "artificial intelligence",
            "max_web_results": 2,
            "max_academic_results": 2,
            "include_arxiv": True,
            "include_semantic_scholar": True,
            "include_web": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "artificial intelligence"
    assert "mocked data" in data["summary"]
    assert len(data["academic_hits"]) >= 2
    assert "Mocked Paper 1" in [hit["title"] for hit in data["academic_hits"]]
    assert "/mock/path/arxiv.md" in data["arxiv_files"]


@patch("services.citation_service.LLM")
@patch("services.citation_service.get_paper_citations")
@patch("services.citation_service.get_paper_references")
@patch("services.citation_service.get_paper")
@patch("services.citation_service.get_paper_id")
def test_citations_endpoint(mock_get_id, mock_get_paper, mock_get_refs, mock_get_cits, mock_llm):
    """Test the /citations/ endpoint with mocked Semantic Scholar graph and LLM responses."""
    mock_get_id.return_value = "mock_paper_id_123"
    
    mock_get_paper.return_value = {
        "paperId": "mock_paper_id_123",
        "title": "Canonical Mock Paper",
        "year": 2025,
        "authors": [{"name": "Author A"}],
        "abstract": "Abstract of canonical paper",
        "citationCount": 42,
        "openAccessPdf": {"url": "http://example.com/canonical.pdf"}
    }
    
    mock_get_refs.return_value = {
        "data": [
            {
                "citedPaper": {
                    "paperId": "ref_id_1",
                    "title": "Reference Paper 1",
                    "year": 2020,
                    "authors": [{"name": "Author B"}],
                    "abstract": "Abstract of reference paper",
                    "citationCount": 5,
                    "openAccessPdf": {"url": "http://example.com/ref1.pdf"}
                }
            }
        ]
    }
    
    mock_get_cits.return_value = {
        "data": [
            {
                "citingPaper": {
                    "paperId": "cit_id_1",
                    "title": "Citing Paper 1",
                    "year": 2026,
                    "authors": [{"name": "Author C"}],
                    "abstract": "Abstract of citing paper",
                    "citationCount": 1,
                    "openAccessPdf": {"url": "http://example.com/cit1.pdf"}
                }
            }
        ]
    }
    
    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is a generated citation landscape brief."
    mock_llm.invoke.return_value = mock_llm_response

    response = client.post(
        "/citations/",
        json={
            "query": "Canonical Mock Paper",
            "limit": 5,
            "include_references": True,
            "include_citations": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Canonical Mock Paper"
    assert data["resolved_paper"]["title"] == "Canonical Mock Paper"
    assert len(data["references"]) == 1
    assert data["references"][0]["title"] == "Reference Paper 1"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["title"] == "Citing Paper 1"
    assert "citation landscape brief" in data["context_summary"]


@patch("services.literature_service.LLM")
@patch("services.literature_service.paper_search")
@patch("services.literature_service.fetch_and_save_best_arxiv_paper")
@patch("services.literature_service.query_db")
def test_literature_review_endpoint(mock_query_db, mock_arxiv_search, mock_ss_search, mock_llm):
    """Test the /literature/review endpoint with mocked Vector DB search, external fallback and LLM responses."""
    # Mocking Vector DB search returning no items, triggering fallback
    mock_query_db.return_value = []
    
    # Mocking fallback arXiv and Semantic Scholar
    mock_arxiv_search.return_value = {
        "saved_papers": [
            {
                "title": "Supplementary arXiv Paper",
                "file_path": "/mock/path/arxiv2.md"
            }
        ]
    }
    
    mock_ss_search.return_value = [
        {
            "title": "Supplementary Semantic Scholar Paper",
            "abstract": "Abstract of supplementary paper",
            "year": 2025,
            "citations": 8,
            "pdf": "http://example.com/ss_supplementary.pdf"
        }
    ]
    
    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is a generated structured literature review."
    mock_llm.invoke.return_value = mock_llm_response

    response = client.post(
        "/literature/review",
        json={
            "topic": "deep learning",
            "n_db_results": 5,
            "min_hits": 2,
            "supplement_with_web": True,
            "supplement_with_arxiv": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "deep learning"
    assert len(data["db_chunks"]) == 0
    assert len(data["supplementary_papers"]) == 2
    assert "Supplementary arXiv Paper" in [p["title"] for p in data["supplementary_papers"]]
    assert "Supplementary Semantic Scholar Paper" in [p["title"] for p in data["supplementary_papers"]]
    assert "structured literature review" in data["review"]
