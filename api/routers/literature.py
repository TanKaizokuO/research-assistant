"""
api/routers/literature.py — Literature Review router.

POST /literature/ingest
    Body: multipart/form-data  (files[])
    Response: {"queued": [...], "message": "..."}

POST /literature/review
    Body: LiteratureRequest
    Response: LiteratureResponse
"""
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from api.dependencies import get_tavily_key
from services.literature_service import (
    LiteratureRequest,
    LiteratureResponse,
    ingest_uploaded_pdfs,
    run_literature_review,
)

router = APIRouter()


@router.post("/ingest", summary="Upload PDFs for ingestion")
async def ingest_pdfs_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """Save uploaded PDF files and trigger ChromaDB ingestion in the background.

    The HTTP response is returned immediately. The embedding step (which can
    be slow on first run while the BGE model is downloaded) runs in a
    BackgroundTask and does not block the caller.
    """
    return await ingest_uploaded_pdfs(files, background_tasks)


@router.post("/review", response_model=LiteratureResponse, summary="Generate a literature review")
async def literature_review(
    req: LiteratureRequest,
    tavily_key: str = Depends(get_tavily_key),
):
    """Query the vector DB for relevant chunks, optionally supplement with
    arXiv / Semantic Scholar results, and generate a structured literature
    review using the NVIDIA LLM."""
    return await run_literature_review(req, tavily_key)
