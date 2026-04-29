"""
api/routers/citations.py — Citation & Reference Finder router.

POST /citations/
    Body: CitationRequest
    Response: CitationResponse
"""
from fastapi import APIRouter

from services.citation_service import CitationRequest, CitationResponse, run_citation_finder

router = APIRouter()


@router.post("/", response_model=CitationResponse, summary="Find citations for a paper")
async def find_citations(req: CitationRequest):
    """Resolve a paper on Semantic Scholar, fetch its references and citing
    papers, and return an LLM-generated citation-landscape summary."""
    return await run_citation_finder(req)
