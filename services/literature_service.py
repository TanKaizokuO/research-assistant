"""
services/literature_service.py — Literature Review orchestrator.

Handles PDF ingestion (save + background ChromaDB embedding) and
literature review generation (vector DB query + optional supplementation
from arXiv / Semantic Scholar + LLM prose generation).
"""
import asyncio
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from data_loaders.arxiv_loader import fetch_and_save_best_arxiv_paper
from data_loaders.pdf_ingestion import ingest_pdfs, query_db
from data_loaders.semantic_scholar_loader import paper_search
from services import LLM
from services.research_service import AcademicHit  # reuse shared model

# ── Directory constants ────────────────────────────────────────────────────

PDF_DIR = Path("pdf-from-user")
DB_DIR = Path("chroma_db")
COLLECTION = "literature_db"


# ── Request / Response models ──────────────────────────────────────────────


class LiteratureRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500, description="Research topic or question")
    n_db_results: int = Field(default=10, ge=1, le=50, description="Number of ChromaDB results to retrieve")
    min_hits: int = Field(default=3, ge=1, le=10, description="Supplement if fewer than this many DB hits are returned")
    supplement_with_web: bool = Field(default=True, description="Supplement with Semantic Scholar if DB hits are low")
    supplement_with_arxiv: bool = Field(default=True, description="Supplement with arXiv if DB hits are low")


class LiteratureChunk(BaseModel):
    text: str
    score: float
    title: str
    authors: str
    year: str
    doi: str
    filename: str


class LiteratureResponse(BaseModel):
    topic: str
    review: str = Field(description="LLM-generated structured literature review")
    db_chunks: List[LiteratureChunk] = Field(description="Relevant chunks from ChromaDB")
    supplementary_papers: List[AcademicHit] = Field(description="Papers from arXiv / Semantic Scholar when DB hits are low")
    errors: List[str]


# ── Ingestion helpers ──────────────────────────────────────────────────────


async def ingest_uploaded_pdfs(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
) -> dict:
    """Save uploaded PDF files to disk, then schedule ChromaDB ingestion as a
    BackgroundTask so the HTTP response is returned immediately.

    Non-PDF files are silently skipped (not saved, not reported as errors).
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for upload in files:
        if not (upload.filename or "").lower().endswith(".pdf"):
            continue  # skip non-PDFs silently
        dest = PDF_DIR / upload.filename
        with open(dest, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved.append(upload.filename)

    # Heavy embedding work runs after the response has been sent
    background_tasks.add_task(_run_ingest_sync)

    return {"queued": saved, "message": "Ingestion started in background"}


def _run_ingest_sync() -> None:
    """Synchronous ChromaDB ingestion — safe to call from BackgroundTasks."""
    ingest_pdfs(pdf_dir=PDF_DIR, db_dir=DB_DIR, collection=COLLECTION, batch_size=64)


# ── Literature review ──────────────────────────────────────────────────────


async def run_literature_review(req: LiteratureRequest, tavily_key: str) -> LiteratureResponse:
    """Generate a structured literature review.

    Steps:
    1. Query ChromaDB for semantically relevant chunks.
    2. If fewer than min_hits chunks are returned, supplement from arXiv
       and/or Semantic Scholar concurrently.
    3. Build an LLM prompt and generate academic prose.
    """
    errors: List[str] = []

    # ── Step 1: Query ChromaDB ─────────────────────────────────────────────
    try:
        raw_hits = await asyncio.to_thread(
            query_db, req.topic, req.n_db_results, DB_DIR, COLLECTION
        )
    except Exception as e:
        raw_hits = []
        errors.append(f"Vector DB query failed: {e}")

    db_chunks: List[LiteratureChunk] = [
        LiteratureChunk(
            text=h["text"],
            score=h["score"],
            title=h["metadata"].get("title", ""),
            authors=h["metadata"].get("authors", ""),
            year=str(h["metadata"].get("year", "")),
            doi=h["metadata"].get("doi", ""),
            filename=h["metadata"].get("filename", ""),
        )
        for h in raw_hits
    ]

    # ── Step 2: Supplement if DB hits are sparse ───────────────────────────
    supplementary: List[AcademicHit] = []

    if len(db_chunks) < req.min_hits:
        supplement_tasks: dict[str, asyncio.Future] = {}

        if req.supplement_with_arxiv:
            supplement_tasks["arxiv"] = asyncio.to_thread(
                fetch_and_save_best_arxiv_paper, req.topic
            )
        if req.supplement_with_web:  # uses Semantic Scholar for academic supplement
            supplement_tasks["ss"] = asyncio.to_thread(paper_search, req.topic, 5)

        if supplement_tasks:
            sup_results = await asyncio.gather(
                *supplement_tasks.values(), return_exceptions=True
            )
            sup_map = dict(zip(supplement_tasks.keys(), sup_results))

            if "arxiv" in sup_map and not isinstance(sup_map["arxiv"], Exception):
                for p in sup_map["arxiv"].get("saved_papers", []):
                    supplementary.append(
                        AcademicHit(
                            title=p["title"],
                            abstract=None,
                            year=None,
                            citations=None,
                            pdf_url=None,
                            source="arxiv",
                        )
                    )

            if "ss" in sup_map and not isinstance(sup_map["ss"], Exception):
                for p in sup_map["ss"]:
                    supplementary.append(
                        AcademicHit(
                            title=p.get("title") or "Unknown",
                            abstract=p.get("abstract"),
                            year=p.get("year"),
                            citations=p.get("citations"),
                            pdf_url=p.get("pdf"),
                            source="semantic_scholar",
                        )
                    )

    # ── Step 3: Build LLM context ──────────────────────────────────────────
    chunk_context = "\n\n".join(
        [
            f"[From: {c.title or c.filename}, {c.year}]\n{c.text[:600]}"
            for c in db_chunks[:8]
        ]
    )
    sup_context = "\n".join(
        [
            f"- {a.title} ({a.year or 'n/d'}): {(a.abstract or '')[:300]}"
            for a in supplementary[:5]
        ]
    )

    prompt = (
        f'You are an expert academic writer. Write a structured literature review on:\n"{req.topic}"\n\n'
        "Use ONLY the content below. Structure your review with:\n"
        "1. Introduction\n"
        "2. Key Themes & Findings\n"
        "3. Methodological Approaches (if relevant)\n"
        "4. Gaps & Future Directions\n"
        "5. Conclusion\n\n"
        "---\n"
        f"PRIMARY SOURCES (from user PDFs):\n{chunk_context or 'None provided'}\n\n"
        f"SUPPLEMENTARY SOURCES:\n{sup_context or 'None'}\n"
        "---\n\n"
        "Do not invent citations. Use author names and years where available."
    )

    try:
        llm_response = await asyncio.to_thread(LLM.invoke, [HumanMessage(content=prompt)])
        review = llm_response.content.strip()
    except Exception as e:
        review = "Literature review generation failed."
        errors.append(f"LLM generation failed: {e}")

    return LiteratureResponse(
        topic=req.topic,
        review=review,
        db_chunks=db_chunks,
        supplementary_papers=supplementary,
        errors=errors,
    )
