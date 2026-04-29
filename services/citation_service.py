"""
services/citation_service.py — Citation & Reference Finder orchestrator.

Resolves a paper on Semantic Scholar, fetches its references and citing
papers concurrently, and generates an LLM citation-landscape summary.
"""
import asyncio
from typing import List, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from data_loaders.semantic_scholar_loader import (
    get_paper,
    get_paper_citations,
    get_paper_id,
    get_paper_references,
)
from services import LLM

# Fields requested from the Semantic Scholar API
PAPER_FIELDS = "title,abstract,year,authors,citationCount,openAccessPdf,externalIds"
REF_FIELDS = "title,abstract,year,authors,citationCount,openAccessPdf"


# ── Request / Response models ──────────────────────────────────────────────


class CitationRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Paper title, DOI, or research topic to look up",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max references / citations to return")
    include_references: bool = Field(default=True, description="Include papers this paper cites")
    include_citations: bool = Field(default=True, description="Include papers that cite this paper")


class PaperRef(BaseModel):
    paperId: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    citation_count: Optional[int] = None
    open_access_pdf: Optional[str] = None


class CitationResponse(BaseModel):
    query: str
    resolved_paper: Optional[PaperRef] = None
    references: List[PaperRef] = Field(description="Papers this paper cites")
    citations: List[PaperRef] = Field(description="Papers that cite this paper")
    context_summary: str = Field(description="LLM-generated citation-landscape summary")
    errors: List[str]


# ── Parsing helpers ────────────────────────────────────────────────────────


def _parse_paper(raw: dict) -> PaperRef:
    """Convert a raw Semantic Scholar API paper dict into a PaperRef model."""
    authors = [a.get("name", "") for a in raw.get("authors", [])]
    pdf = (raw.get("openAccessPdf") or {}).get("url")
    return PaperRef(
        paperId=raw.get("paperId"),
        title=raw.get("title"),
        year=raw.get("year"),
        authors=authors,
        abstract=raw.get("abstract"),
        citation_count=raw.get("citationCount"),
        open_access_pdf=pdf,
    )


def _parse_edge_list(raw: dict, edge_key: str) -> List[PaperRef]:
    """Parse Semantic Scholar graph edges.

    The API returns `{"data": [{edge_key: {paper fields}}, ...]}`.
    `edge_key` is "citedPaper" for references and "citingPaper" for citations.
    """
    results: List[PaperRef] = []
    for item in raw.get("data", []):
        paper_data = item.get(edge_key) or {}
        if paper_data:
            results.append(_parse_paper(paper_data))
    return results


# ── Service function ───────────────────────────────────────────────────────


async def run_citation_finder(req: CitationRequest) -> CitationResponse:
    """Resolve a paper on Semantic Scholar, fetch its reference/citation
    graph concurrently, and return an LLM-generated landscape summary.

    If the paper ID cannot be resolved, returns immediately with an error
    and empty lists rather than raising an HTTP 500.
    """
    errors: List[str] = []

    # ── Step 1: Resolve canonical paper ID ────────────────────────────────
    try:
        paper_id = await asyncio.to_thread(get_paper_id, req.query)
    except Exception as e:
        errors.append(f"Could not resolve paper ID: {e}")
        return CitationResponse(
            query=req.query,
            resolved_paper=None,
            references=[],
            citations=[],
            context_summary="Could not resolve the paper.",
            errors=errors,
        )

    # ── Step 2: Fetch details + references + citations concurrently ────────
    tasks: List[asyncio.Future] = [
        asyncio.to_thread(get_paper, paper_id, PAPER_FIELDS),
    ]

    if req.include_references:
        tasks.append(
            asyncio.to_thread(get_paper_references, paper_id, REF_FIELDS, req.limit, 0)
        )
    if req.include_citations:
        tasks.append(
            asyncio.to_thread(get_paper_citations, paper_id, REF_FIELDS, req.limit, 0)
        )

    fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results in insertion order
    idx = 0
    resolved_paper: Optional[PaperRef] = None
    references: List[PaperRef] = []
    citations_list: List[PaperRef] = []

    paper_raw = fetch_results[idx]; idx += 1
    if isinstance(paper_raw, Exception):
        errors.append(f"Paper detail fetch failed: {paper_raw}")
    else:
        resolved_paper = _parse_paper(paper_raw)

    if req.include_references:
        refs_raw = fetch_results[idx]; idx += 1
        if isinstance(refs_raw, Exception):
            errors.append(f"References fetch failed: {refs_raw}")
        else:
            references = _parse_edge_list(refs_raw, "citedPaper")

    if req.include_citations:
        cits_raw = fetch_results[idx]; idx += 1
        if isinstance(cits_raw, Exception):
            errors.append(f"Citations fetch failed: {cits_raw}")
        else:
            citations_list = _parse_edge_list(cits_raw, "citingPaper")

    # ── Step 3: LLM citation-landscape summary ────────────────────────────
    ref_titles = "\n".join(
        [f"- {r.title} ({r.year})" for r in references[:10] if r.title]
    )
    cit_titles = "\n".join(
        [f"- {c.title} ({c.year})" for c in citations_list[:10] if c.title]
    )
    paper_abstract = (resolved_paper.abstract or "") if resolved_paper else ""

    prompt = (
        "You are a research assistant. Summarize the citation landscape for this paper:\n\n"
        f"Title: {resolved_paper.title if resolved_paper else req.query}\n"
        f"Abstract: {paper_abstract[:500] if paper_abstract else 'N/A'}\n\n"
        f"Papers it references (what it builds on):\n{ref_titles or 'None found'}\n\n"
        f"Papers that cite it (its impact):\n{cit_titles or 'None found'}\n\n"
        "Write 2-3 paragraphs covering:\n"
        "1. What foundational work this paper builds on\n"
        "2. The research areas influenced by this paper\n"
        "3. Its overall significance"
    )

    try:
        llm_resp = await asyncio.to_thread(LLM.invoke, [HumanMessage(content=prompt)])
        context_summary = llm_resp.content.strip()
    except Exception as e:
        context_summary = "Summary generation failed."
        errors.append(f"LLM summary failed: {e}")

    return CitationResponse(
        query=req.query,
        resolved_paper=resolved_paper,
        references=references,
        citations=citations_list,
        context_summary=context_summary,
        errors=errors,
    )
