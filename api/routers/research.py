"""
api/routers/research.py — Topic Research router.

POST /research/
    Body: ResearchRequest
    Response: ResearchResponse
"""
from fastapi import APIRouter, Depends

from api.dependencies import get_tavily_key
from services.research_service import ResearchRequest, ResearchResponse, run_research

router = APIRouter()


@router.post("/", response_model=ResearchResponse, summary="Research a topic")
async def research_topic(
    req: ResearchRequest,
    tavily_key: str = Depends(get_tavily_key),
):
    """Fan out to web, arXiv, and Semantic Scholar in parallel, then generate
    an LLM research brief from the merged results."""
    return await run_research(req, tavily_key)
