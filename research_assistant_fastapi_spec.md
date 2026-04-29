# Research Assistant – FastAPI Backend Specification

> **Audience:** Coding agent implementing the backend from scratch.
> **Goal:** A production-ready FastAPI backend that exposes three core features — Topic Research, Literature Review (with optional user-supplied PDFs), and Citation/Reference Finder — built on top of the existing `data_loaders` and `output_schemas` modules.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment & Dependencies](#2-environment--dependencies)
3. [Existing Modules — Contracts & Known Bugs](#3-existing-modules--contracts--known-bugs)
4. [FastAPI Application](#4-fastapi-application)
5. [Feature 1 — Topic Research](#5-feature-1--topic-research)
6. [Feature 2 — Literature Review](#6-feature-2--literature-review)
7. [Feature 3 — Citation & Reference Finder](#7-feature-3--citation--reference-finder)
8. [Shared Utilities](#8-shared-utilities)
9. [Error Handling](#9-error-handling)
10. [Configuration & Secrets](#10-configuration--secrets)
11. [main.py Fix](#11-mainpy-fix)

---

## 1. Project Structure

```
research_assistant/
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI app factory
│   ├── dependencies.py         # Shared FastAPI dependencies (API keys, clients)
│   └── routers/
│       ├── __init__.py
│       ├── research.py         # Feature 1 – Topic Research
│       ├── literature.py       # Feature 2 – Literature Review
│       └── citations.py        # Feature 3 – Citation Finder
├── data_loaders/
│   ├── arxiv_loader.py         # (existing — do NOT modify)
│   ├── pdf_ingestion.py        # (existing — do NOT modify)
│   ├── semantic_scholar_loader.py  # (existing — do NOT modify)
│   └── web_search.py           # (existing — do NOT modify)
├── output_schemas/
│   └── schema.py               # (existing — do NOT modify)
├── services/
│   ├── __init__.py
│   ├── research_service.py     # Orchestrates Feature 1
│   ├── literature_service.py   # Orchestrates Feature 2
│   └── citation_service.py     # Orchestrates Feature 3
├── logger.py                   # (existing — do NOT modify)
├── main.py                     # Fixed entry point (see §11)
├── .env                        # Secrets (never commit)
└── requirements.txt
```

---

## 2. Environment & Dependencies

### `.env` (required keys)

```dotenv
TAVILY_API_KEY=tvly-...
NVIDIA_API_KEY=nvapi-...
SEMANTIC_SCHOLAR_API_KEY=...   # optional — leave blank for unauthenticated access
```

### `requirements.txt` additions (on top of existing deps)

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9        # for UploadFile support
httpx>=0.27.0
pydantic>=2.7.0
python-dotenv>=1.0.0
```

---

## 3. Existing Modules — Contracts & Known Bugs

The coding agent must understand the existing module interfaces **without modifying them**. All bugs below must be worked around inside the new service layer.

### 3.1 `data_loaders/arxiv_loader.py`

**Public function:**
```python
fetch_and_save_best_arxiv_paper(query: str) -> dict
# Returns: {"selected_indices": [int], "saved_papers": [{"index": int, "title": str, "file_path": str}]}
```
- Side-effect: writes `.md` files to `./arxiv/` directory relative to CWD.
- The function instantiates `ChatNVIDIA` at module load time using `NVIDIA_API_KEY` from env.
- No known blocking bugs — import and call directly.

### 3.2 `data_loaders/pdf_ingestion.py`

**Public functions:**
```python
ingest_pdfs(pdf_dir, db_dir, collection, batch_size, force) -> dict
# Returns: {"status": str, "ingested": int, "skipped": int, "errors": [str], "total_chunks": int, ...}

query_db(query, n_results, db_dir, collection) -> list[dict]
# Returns: [{"text": str, "score": float, "metadata": dict}, ...]
```
- PDFs must be placed in `pdf_dir` before calling `ingest_pdfs`.
- `BGEEmbedder` downloads the `BAAI/bge-base-en` model on first call — this can be slow; the service layer should run it in a background task.
- No known blocking bugs.

### 3.3 `data_loaders/semantic_scholar_loader.py`

**Public functions:**
```python
paper_search(query: str, limit: int = 5) -> list[dict]
# Returns: [{"paperId", "title", "abstract", "year", "citations", "pdf"}, ...]

snippet_search(query: str, limit: int = 3) -> list[dict]
# Returns: [{"text", "paperId", "title"}, ...]

get_paper_id(paper_id: str) -> str
# Performs fuzzy match — returns the canonical Semantic Scholar paperId string

get_paper(paper_id: str, fields: str) -> dict
get_paper_references(paper_id: str, fields, limit, offset) -> dict
get_paper_citations(paper_id: str, fields, limit, offset) -> dict
```
- All functions use `requests` (blocking). Wrap in `asyncio.to_thread()` inside async route handlers.
- `SEMANTIC_SCHOLAR_API_KEY` is optional; without it, requests are rate-limited more aggressively.

### 3.4 `data_loaders/web_search.py`

**Public function:**
```python
manual_web_search(tavily_api_key, query, max_results, nvidia_api_key) -> List[SourceSchema]
```
- Uses `requests` (blocking). Wrap in `asyncio.to_thread()`.
- **Known bug:** `confidence_score` logic is inverted — when extraction succeeds, the model scores it correctly, but when fallback is used, it hard-codes `75` instead of using the LLM score. **Do not fix in the module.** Accept the output as-is.

### 3.5 `output_schemas/schema.py`

```python
class SourceSchema(BaseModel): ...   # summarised source
class DataSchema(BaseModel): ...     # full-content source
```
- `SourceSchema.summary` max 2000 chars. Already validated by the module.
- `DataSchema.url` is `Optional[HttpUrl]`.

### 3.6 `main.py` — Bug

The existing `main.py` has two bugs (see §11 for the fix):
1. Imports `Tavily_Client` which does not exist — correct name is `manual_web_search`.
2. Calls `from numpy import rint` unnecessarily.

---

## 4. FastAPI Application

### `api/app.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import research, literature, citations

def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Assistant API",
        version="1.0.0",
        description="AI-powered research, literature review, and citation finder",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research.router,    prefix="/research",    tags=["Topic Research"])
    app.include_router(literature.router,  prefix="/literature",  tags=["Literature Review"])
    app.include_router(citations.router,   prefix="/citations",   tags=["Citations"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app

app = create_app()
```

### `api/dependencies.py`

```python
import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

@lru_cache()
def get_tavily_key() -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set")
    return key

@lru_cache()
def get_nvidia_key() -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key
```

---

## 5. Feature 1 — Topic Research

### What it does

Given a free-text research topic, fan out to **three sources in parallel**:
1. **Web** via Tavily (`web_search.manual_web_search`)
2. **arXiv** (`arxiv_loader.fetch_and_save_best_arxiv_paper`)
3. **Semantic Scholar** (`semantic_scholar_loader.paper_search`)

Merge and rank all results, then generate a structured research brief using the NVIDIA LLM.

---

### Request / Response Models

```python
# services/research_service.py (Pydantic models)

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    max_web_results: int = Field(default=5, ge=1, le=20)
    max_academic_results: int = Field(default=5, ge=1, le=10)
    include_arxiv: bool = True
    include_semantic_scholar: bool = True
    include_web: bool = True

class AcademicHit(BaseModel):
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citations: Optional[int]
    pdf_url: Optional[str]
    source: str  # "arxiv" | "semantic_scholar"

class ResearchResponse(BaseModel):
    topic: str
    summary: str                      # LLM-generated research brief (500-2000 chars)
    web_sources: List[SourceSchema]
    academic_hits: List[AcademicHit]
    arxiv_files: List[str]            # local file paths of saved .md files
    errors: List[str]                 # non-fatal per-source errors
```

---

### Router — `api/routers/research.py`

```python
from fastapi import APIRouter, Depends
from api.dependencies import get_tavily_key
from services.research_service import ResearchRequest, ResearchResponse, run_research

router = APIRouter()

@router.post("/", response_model=ResearchResponse)
async def research_topic(req: ResearchRequest, tavily_key: str = Depends(get_tavily_key)):
    return await run_research(req, tavily_key)
```

---

### Service — `services/research_service.py`

```python
import asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from data_loaders.web_search import manual_web_search
from data_loaders.arxiv_loader import fetch_and_save_best_arxiv_paper
from data_loaders.semantic_scholar_loader import paper_search

LLM = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")

async def run_research(req: ResearchRequest, tavily_key: str) -> ResearchResponse:
    errors = []

    # ── Fan-out: run all sources concurrently ──────────────────────────────
    tasks = {}

    if req.include_web:
        tasks["web"] = asyncio.to_thread(
            manual_web_search, tavily_key, req.topic, req.max_web_results
        )
    if req.include_arxiv:
        tasks["arxiv"] = asyncio.to_thread(
            fetch_and_save_best_arxiv_paper, req.topic
        )
    if req.include_semantic_scholar:
        tasks["ss"] = asyncio.to_thread(
            paper_search, req.topic, req.max_academic_results
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_map = dict(zip(tasks.keys(), results))

    # ── Unpack results safely ──────────────────────────────────────────────
    web_sources = []
    academic_hits = []
    arxiv_files = []

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
                academic_hits.append(AcademicHit(
                    title=p["title"], abstract=None,
                    year=None, citations=None, pdf_url=None, source="arxiv"
                ))

    if "ss" in result_map:
        r = result_map["ss"]
        if isinstance(r, Exception):
            errors.append(f"Semantic Scholar failed: {r}")
        else:
            for p in r:
                academic_hits.append(AcademicHit(
                    title=p["title"] or "Unknown",
                    abstract=p.get("abstract"),
                    year=p.get("year"),
                    citations=p.get("citations"),
                    pdf_url=p.get("pdf"),
                    source="semantic_scholar",
                ))

    # ── Generate research brief ────────────────────────────────────────────
    context_snippets = "\n\n".join(
        [f"- {s.title}: {s.summary[:300]}" for s in web_sources[:5]]
        + [f"- [{a.source}] {a.title}: {a.abstract[:300] if a.abstract else 'No abstract'}"
           for a in academic_hits[:5]]
    )

    prompt = f"""You are a research assistant. Based on the following sources, write a concise 
research brief (500-800 words) about: "{req.topic}"

Sources:
{context_snippets}

Include: key findings, major themes, and open questions. Do not fabricate facts not present in the sources."""

    try:
        llm_response = await asyncio.to_thread(
            LLM.invoke, [HumanMessage(content=prompt)]
        )
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
```

---

## 6. Feature 2 — Literature Review

### What it does

1. **Accepts optional PDF uploads** — saves them to `pdf-from-user/`, then ingests them into ChromaDB via `pdf_ingestion.ingest_pdfs`.
2. **Queries the vector DB** for the given topic using `pdf_ingestion.query_db`.
3. **Supplements** with Semantic Scholar and arXiv results if the DB returns fewer than `min_hits` results.
4. **Generates a literature review** (structured academic prose) via the LLM.

---

### Endpoints

#### `POST /literature/ingest` — Upload PDFs

```
Content-Type: multipart/form-data
Body: files[] — one or more PDF files
```

```python
# api/routers/literature.py

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from services.literature_service import ingest_uploaded_pdfs, LiteratureRequest, run_literature_review

router = APIRouter()

@router.post("/ingest")
async def ingest_pdfs_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Save uploaded PDFs and trigger background ingestion."""
    result = await ingest_uploaded_pdfs(files, background_tasks)
    return result

@router.post("/review", response_model=LiteratureResponse)
async def literature_review(req: LiteratureRequest, tavily_key: str = Depends(get_tavily_key)):
    return await run_literature_review(req, tavily_key)
```

---

### Request / Response Models

```python
class LiteratureRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    n_db_results: int = Field(default=10, ge=1, le=50)
    min_hits: int = Field(default=3, ge=1, le=10)   # supplement if fewer DB hits
    supplement_with_web: bool = True
    supplement_with_arxiv: bool = True

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
    review: str                          # LLM-generated literature review
    db_chunks: List[LiteratureChunk]     # relevant chunks from ChromaDB
    supplementary_papers: List[AcademicHit]  # from arXiv / SS if needed
    errors: List[str]
```

---

### Service — `services/literature_service.py`

```python
import shutil
import asyncio
from pathlib import Path
from fastapi import UploadFile, BackgroundTasks
from data_loaders.pdf_ingestion import ingest_pdfs, query_db
from data_loaders.arxiv_loader import fetch_and_save_best_arxiv_paper
from data_loaders.semantic_scholar_loader import paper_search

PDF_DIR = Path("pdf-from-user")
DB_DIR = Path("chroma_db")
COLLECTION = "literature_db"


async def ingest_uploaded_pdfs(files: List[UploadFile], background_tasks: BackgroundTasks) -> dict:
    """Save uploaded files then run ingestion in the background."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            continue   # silently skip non-PDFs
        dest = PDF_DIR / upload.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(upload.filename)

    # Run the heavy embedding work in a background task so the HTTP response
    # returns immediately. The client should poll /literature/status or just
    # wait before calling /literature/review.
    background_tasks.add_task(_run_ingest_sync)

    return {"queued": saved, "message": "Ingestion started in background"}


def _run_ingest_sync():
    """Synchronous ingestion — called from BackgroundTasks."""
    ingest_pdfs(pdf_dir=PDF_DIR, db_dir=DB_DIR, collection=COLLECTION, batch_size=64)


async def run_literature_review(req: LiteratureRequest, tavily_key: str) -> LiteratureResponse:
    errors = []

    # ── Query ChromaDB ─────────────────────────────────────────────────────
    try:
        raw_hits = await asyncio.to_thread(
            query_db, req.topic, req.n_db_results, DB_DIR, COLLECTION
        )
    except Exception as e:
        raw_hits = []
        errors.append(f"Vector DB query failed: {e}")

    db_chunks = [
        LiteratureChunk(
            text=h["text"],
            score=h["score"],
            title=h["metadata"].get("title", ""),
            authors=h["metadata"].get("authors", ""),
            year=h["metadata"].get("year", ""),
            doi=h["metadata"].get("doi", ""),
            filename=h["metadata"].get("filename", ""),
        )
        for h in raw_hits
    ]

    # ── Supplement if needed ───────────────────────────────────────────────
    supplementary: List[AcademicHit] = []

    if len(db_chunks) < req.min_hits:
        supplement_tasks = {}
        if req.supplement_with_arxiv:
            supplement_tasks["arxiv"] = asyncio.to_thread(
                fetch_and_save_best_arxiv_paper, req.topic
            )
        if req.supplement_with_web:   # reuse SS for academic supplement
            supplement_tasks["ss"] = asyncio.to_thread(paper_search, req.topic, 5)

        if supplement_tasks:
            sup_results = await asyncio.gather(*supplement_tasks.values(), return_exceptions=True)
            sup_map = dict(zip(supplement_tasks.keys(), sup_results))

            if "arxiv" in sup_map and not isinstance(sup_map["arxiv"], Exception):
                for p in sup_map["arxiv"].get("saved_papers", []):
                    supplementary.append(AcademicHit(
                        title=p["title"], abstract=None, year=None,
                        citations=None, pdf_url=None, source="arxiv"
                    ))

            if "ss" in sup_map and not isinstance(sup_map["ss"], Exception):
                for p in sup_map["ss"]:
                    supplementary.append(AcademicHit(
                        title=p["title"] or "Unknown",
                        abstract=p.get("abstract"),
                        year=p.get("year"),
                        citations=p.get("citations"),
                        pdf_url=p.get("pdf"),
                        source="semantic_scholar",
                    ))

    # ── Build LLM context ──────────────────────────────────────────────────
    chunk_context = "\n\n".join(
        [f"[From: {c.title or c.filename}, {c.year}]\n{c.text[:600]}"
         for c in db_chunks[:8]]
    )
    sup_context = "\n".join(
        [f"- {a.title} ({a.year or 'n/d'}): {(a.abstract or '')[:300]}"
         for a in supplementary[:5]]
    )

    prompt = f"""You are an expert academic writer. Write a structured literature review on:
"{req.topic}"

Use ONLY the content below. Structure your review with:
1. Introduction
2. Key Themes & Findings
3. Methodological Approaches (if relevant)
4. Gaps & Future Directions
5. Conclusion

---
PRIMARY SOURCES (from user PDFs):
{chunk_context if chunk_context else "None provided"}

SUPPLEMENTARY SOURCES:
{sup_context if sup_context else "None"}
---

Do not invent citations. Use author names and years where available."""

    try:
        llm_response = await asyncio.to_thread(
            LLM.invoke, [HumanMessage(content=prompt)]
        )
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
```

---

## 7. Feature 3 — Citation & Reference Finder

### What it does

Given a **paper title or topic**, the service:
1. Resolves the canonical Semantic Scholar `paperId` using `get_paper_id`.
2. Fetches full paper details via `get_paper`.
3. Fetches **references** (papers this paper cites) via `get_paper_references`.
4. Fetches **citations** (papers that cite this paper) via `get_paper_citations`.
5. Returns a structured response with both lists, and an LLM-generated citation context summary.

---

### Request / Response Models

```python
class CitationRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300,
                       description="Paper title, DOI, or topic to look up")
    limit: int = Field(default=20, ge=1, le=100)
    include_references: bool = True   # papers this paper cites
    include_citations: bool = True    # papers that cite this paper

class PaperRef(BaseModel):
    paperId: Optional[str]
    title: Optional[str]
    year: Optional[int]
    authors: Optional[List[str]]
    abstract: Optional[str]
    citation_count: Optional[int]
    open_access_pdf: Optional[str]

class CitationResponse(BaseModel):
    query: str
    resolved_paper: Optional[PaperRef]    # the paper that was resolved
    references: List[PaperRef]            # papers it cites
    citations: List[PaperRef]             # papers that cite it
    context_summary: str                  # LLM-generated context
    errors: List[str]
```

---

### Router — `api/routers/citations.py`

```python
from fastapi import APIRouter
from services.citation_service import CitationRequest, CitationResponse, run_citation_finder

router = APIRouter()

@router.post("/", response_model=CitationResponse)
async def find_citations(req: CitationRequest):
    return await run_citation_finder(req)
```

---

### Service — `services/citation_service.py`

```python
import asyncio
from data_loaders.semantic_scholar_loader import (
    get_paper_id, get_paper, get_paper_references, get_paper_citations
)

PAPER_FIELDS = "title,abstract,year,authors,citationCount,openAccessPdf,externalIds"
REF_FIELDS   = "title,abstract,year,authors,citationCount,openAccessPdf"


def _parse_paper(raw: dict) -> PaperRef:
    """Convert raw Semantic Scholar API dict to PaperRef."""
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
    """Parse 'data' -> list of {edge_key: {paper fields}} dicts."""
    results = []
    for item in raw.get("data", []):
        paper_data = item.get(edge_key, {})
        if paper_data:
            results.append(_parse_paper(paper_data))
    return results


async def run_citation_finder(req: CitationRequest) -> CitationResponse:
    errors = []

    # ── Step 1: Resolve paper ID ───────────────────────────────────────────
    try:
        paper_id = await asyncio.to_thread(get_paper_id, req.query)
    except Exception as e:
        errors.append(f"Could not resolve paper ID: {e}")
        return CitationResponse(
            query=req.query, resolved_paper=None,
            references=[], citations=[],
            context_summary="Could not resolve the paper.",
            errors=errors,
        )

    # ── Step 2: Fetch paper details + references + citations concurrently ──
    tasks = [asyncio.to_thread(get_paper, paper_id, PAPER_FIELDS)]

    if req.include_references:
        tasks.append(asyncio.to_thread(
            get_paper_references, paper_id, REF_FIELDS, req.limit, 0
        ))
    if req.include_citations:
        tasks.append(asyncio.to_thread(
            get_paper_citations, paper_id, REF_FIELDS, req.limit, 0
        ))

    fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack in order
    idx = 0
    resolved_paper = None
    references = []
    citations_list = []

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

    # ── Step 3: LLM context summary ───────────────────────────────────────
    ref_titles = "\n".join(
        [f"- {r.title} ({r.year})" for r in references[:10] if r.title]
    )
    cit_titles = "\n".join(
        [f"- {c.title} ({c.year})" for c in citations_list[:10] if c.title]
    )
    paper_abstract = resolved_paper.abstract if resolved_paper else ""

    prompt = f"""You are a research assistant. Summarize the citation landscape for this paper:

Title: {resolved_paper.title if resolved_paper else req.query}
Abstract: {paper_abstract[:500] if paper_abstract else 'N/A'}

Papers it references (what it builds on):
{ref_titles if ref_titles else 'None found'}

Papers that cite it (its impact):
{cit_titles if cit_titles else 'None found'}

Write 2-3 paragraphs covering:
1. What foundational work this paper builds on
2. The research areas influenced by this paper
3. Its overall significance"""

    try:
        llm_resp = await asyncio.to_thread(
            LLM.invoke, [HumanMessage(content=prompt)]
        )
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
```

---

## 8. Shared Utilities

### LLM Singleton

To avoid re-instantiating `ChatNVIDIA` across multiple service files, create a single shared instance:

```python
# services/__init__.py
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

LLM = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")
```

Import in all service files: `from services import LLM`

### Blocking Call Wrapper

All calls to existing `data_loaders` functions use `requests` (synchronous). Always wrap them:

```python
result = await asyncio.to_thread(some_blocking_function, arg1, arg2)
```

Never call blocking functions directly inside `async def` route handlers — this blocks the entire event loop.

---

## 9. Error Handling

### Global Exception Handler

Add to `api/app.py`:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )
```

### Per-Feature Strategy

- All three services collect non-fatal errors in an `errors: List[str]` field and return partial results rather than raising HTTP 500.
- Only raise `HTTPException` for request validation failures (e.g., no PDF uploaded).
- Semantic Scholar rate-limit errors (429) are handled by `rate_limited_get` in the existing module with exponential backoff — no extra wrapping needed.

---

## 10. Configuration & Secrets

All secrets are read from `.env` via `python-dotenv`. The FastAPI dependency functions in `api/dependencies.py` use `@lru_cache()` so keys are read once per process.

Directory paths that the app writes to:

| Path | Created by | Purpose |
|------|-----------|---------|
| `./pdf-from-user/` | `ingest_uploaded_pdfs` | Uploaded PDFs |
| `./chroma_db/` | `pdf_ingestion.ingest_pdfs` | Vector DB |
| `./arxiv/` | `arxiv_loader` | Downloaded arXiv papers |

Ensure the process has write permissions to all three. In Docker, mount them as volumes.

### Running the server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 11. `main.py` Fix

Replace the existing broken `main.py` with:

```python
# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
```

The old `main.py` had two bugs:
1. `from data_loaders.web_search import Tavily_Client` — this export does not exist. The correct callable is `manual_web_search`.
2. `from numpy import rint` — unused import, not installed in the project.

The FastAPI app now serves as the unified entry point; the old direct-run pattern is no longer needed.

---

## Appendix — API Summary

| Method | Path | Feature | Description |
|--------|------|---------|-------------|
| `POST` | `/research/` | Topic Research | Research a topic across web + arXiv + Semantic Scholar |
| `POST` | `/literature/ingest` | Literature Review | Upload PDFs for ingestion |
| `POST` | `/literature/review` | Literature Review | Generate a literature review |
| `POST` | `/citations/` | Citation Finder | Find references and citations for a paper |
| `GET`  | `/health` | — | Health check |
