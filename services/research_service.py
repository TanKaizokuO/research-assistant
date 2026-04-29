"""
services/research_service.py — Topic Research orchestrator.

Fans out to three sources concurrently (web, arXiv, Semantic Scholar),
merges results, and generates an LLM research brief.
"""
import asyncio
from typing import List, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from data_loaders.arxiv_loader import fetch_and_save_best_arxiv_paper
from data_loaders.semantic_scholar_loader import paper_search
from data_loaders.web_search import manual_web_search
from output_schemas.schema import SourceSchema
from services import LLM


# ── Request / Response models ──────────────────────────────────────────────


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500, description="Research topic or question")
    max_web_results: int = Field(default=5, ge=1, le=20, description="Maximum Tavily web results")
    max_academic_results: int = Field(default=5, ge=1, le=10, description="Maximum Semantic Scholar results")
    include_arxiv: bool = Field(default=True, description="Include arXiv paper search")
    include_semantic_scholar: bool = Field(default=True, description="Include Semantic Scholar search")
    include_web: bool = Field(default=True, description="Include Tavily web search")


class AcademicHit(BaseModel):
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    pdf_url: Optional[str] = None
    source: str  # "arxiv" | "semantic_scholar"


class ResearchResponse(BaseModel):
    topic: str
    summary: str = Field(description="LLM-generated research brief (500-800 words)")
    web_sources: List[SourceSchema]
    academic_hits: List[AcademicHit]
    arxiv_files: List[str] = Field(description="Local file paths of saved arXiv .md files")
    errors: List[str] = Field(description="Non-fatal per-source errors")


# ── Service function ───────────────────────────────────────────────────────


async def run_research(req: ResearchRequest, tavily_key: str) -> ResearchResponse:
    """Fan out to all enabled sources concurrently, merge results, and
    produce an LLM-generated research brief.

    Non-fatal errors (e.g. one source timing out) are collected in the
    `errors` list; partial results are still returned to the caller.
    """
    errors: List[str] = []

    # ── Fan-out: launch all enabled sources concurrently ──────────────────
    task_map: dict[str, asyncio.Future] = {}

    if req.include_web:
        # nvidia_api_key is optional in manual_web_search (defaults to None)
        task_map["web"] = asyncio.to_thread(
            manual_web_search,
            tavily_key,
            req.topic,
            req.max_web_results,
        )
    if req.include_arxiv:
        task_map["arxiv"] = asyncio.to_thread(
            fetch_and_save_best_arxiv_paper, req.topic
        )
    if req.include_semantic_scholar:
        task_map["ss"] = asyncio.to_thread(
            paper_search, req.topic, req.max_academic_results
        )

    raw_results = await asyncio.gather(*task_map.values(), return_exceptions=True)
    result_map: dict[str, object] = dict(zip(task_map.keys(), raw_results))

    # ── Unpack results safely ─────────────────────────────────────────────
    web_sources: List[SourceSchema] = []
    academic_hits: List[AcademicHit] = []
    arxiv_files: List[str] = []

    if "web" in result_map:
        r = result_map["web"]
        if isinstance(r, Exception):
            errors.append(f"Web search failed: {r}")
        else:
            web_sources = r  # already List[SourceSchema]

    if "arxiv" in result_map:
        r = result_map["arxiv"]
        if isinstance(r, Exception):
            errors.append(f"arXiv fetch failed: {r}")
        else:
            arxiv_files = [p["file_path"] for p in r.get("saved_papers", [])]
            for p in r.get("saved_papers", []):
                academic_hits.append(
                    AcademicHit(
                        title=p["title"],
                        abstract=None,
                        year=None,
                        citations=None,
                        pdf_url=None,
                        source="arxiv",
                    )
                )

    if "ss" in result_map:
        r = result_map["ss"]
        if isinstance(r, Exception):
            errors.append(f"Semantic Scholar failed: {r}")
        else:
            for p in r:
                academic_hits.append(
                    AcademicHit(
                        title=p.get("title") or "Unknown",
                        abstract=p.get("abstract"),
                        year=p.get("year"),
                        citations=p.get("citations"),
                        pdf_url=p.get("pdf"),
                        source="semantic_scholar",
                    )
                )

    # ── Build LLM prompt from merged context ──────────────────────────────
    context_snippets = "\n\n".join(
        [f"- {s.title}: {s.summary[:300]}" for s in web_sources[:5]]
        + [
            f"- [{a.source}] {a.title}: "
            f"{a.abstract[:300] if a.abstract else 'No abstract'}"
            for a in academic_hits[:5]
        ]
    )

    prompt = (
        f'You are a research assistant. Based on the following sources, write a concise '
        f'research brief (500-800 words) about: "{req.topic}"\n\n'
        f"Sources:\n{context_snippets or 'No sources available.'}\n\n"
        "Include: key findings, major themes, and open questions. "
        "Do not fabricate facts not present in the sources."
    )

    try:
        llm_response = await asyncio.to_thread(LLM.invoke, [HumanMessage(content=prompt)])
        summary = llm_response.content.strip()
    except Exception as e:
        summary = "Summary generation failed."
        errors.append(f"LLM summary failed: {e}")

    return ResearchResponse(
        topic=req.topic,
        summary=summary,
        web_sources=web_sources,
        academic_hits=academic_hits,
        arxiv_files=arxiv_files,
        errors=errors,
    )
