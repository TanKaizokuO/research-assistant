import os
import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR = Path("pdf-from-user")
DB_DIR = Path("vector_db")
COLLECTION = "literature_review"

EMBED_MODEL = "BAAI/bge-small-en"

# Chunking — tuned for academic papers
CHUNK_SIZE = 512  # tokens (approx chars / 4)
CHUNK_OVERLAP = 80  # overlap keeps context across chunk boundaries

# BGE models need this prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class PaperMeta:
    """Metadata extracted from / inferred about a research PDF."""

    filename: str
    title: str = ""
    authors: str = ""
    year: str = ""
    abstract: str = ""
    doi: str = ""
    pages: int = 0
    file_hash: str = ""


@dataclass
class Chunk:
    text: str
    chunk_idx: int
    meta: PaperMeta
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # Stable ID: hash(file + chunk index)
        raw = f"{self.meta.file_hash}::{self.chunk_idx}"
        self.chunk_id = hashlib.md5(raw.encode()).hexdigest()


# ── Helpers ───────────────────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def extract_text_pdfplumber(path: Path) -> list[str]:
    """Return per-page text strings using pdfplumber (best for text PDFs)."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
    except Exception as exc:
        log.warning(
            "pdfplumber failed on %s: %s — falling back to pypdf", path.name, exc
        )
        pages = extract_text_pypdf(path)
    return pages


def extract_text_pypdf(path: Path) -> list[str]:
    """Fallback extractor using pypdf."""
    reader = PdfReader(str(path))
    return [p.extract_text() or "" for p in reader.pages]


def extract_metadata(path: Path) -> PaperMeta:
    """
    Best-effort metadata extraction.
    Pulls PDF document info first, then heuristically scans page-1 text
    for title, authors, year, DOI, and abstract.
    """
    meta = PaperMeta(filename=path.name, file_hash=file_sha256(path))

    # --- PyPDF doc info (often populated in publisher PDFs) ---
    try:
        reader = PdfReader(str(path))
        info = reader.metadata or {}
        meta.title = _clean(info.get("/Title", ""))
        meta.authors = _clean(info.get("/Author", ""))
        meta.pages = len(reader.pages)
    except Exception:
        pass

    # --- Heuristic scan of first two pages ---
    pages = extract_text_pdfplumber(path)
    meta.pages = meta.pages or len(pages)
    first_two = "\n".join(pages[:2]) if pages else ""

    if not meta.title:
        meta.title = _heuristic_title(first_two, path.stem)

    if not meta.authors:
        meta.authors = _heuristic_authors(first_two)

    meta.year = _heuristic_year(first_two)
    meta.doi = _heuristic_doi(first_two)
    meta.abstract = _heuristic_abstract(first_two)

    return meta


# ── Text cleaning & chunking ──────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Remove noise common in PDF-extracted academic text."""
    # Ligature fixes
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    # Collapse excessive whitespace / hyphenation at line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newline → space
    text = re.sub(r"\n{3,}", "\n\n", text)  # ≥3 blank lines → 2
    text = re.sub(r"[ \t]{2,}", " ", text)  # multiple spaces
    return text.strip()


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Sentence-aware sliding-window chunker.
    Splits on sentence boundaries; packs sentences until chunk_size
    (measured in characters, ~tokens*4), then slides by (chunk_size - overlap).
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if buf_len + sent_len > chunk_size * 4 and buf:
            chunks.append(" ".join(buf))
            # keep overlap
            overlap_buf: list[str] = []
            ol = 0
            for s in reversed(buf):
                if ol + len(s) > overlap * 4:
                    break
                overlap_buf.insert(0, s)
                ol += len(s)
            buf = overlap_buf
            buf_len = ol
        buf.append(sent)
        buf_len += sent_len

    if buf:
        chunks.append(" ".join(buf))

    return [c for c in chunks if len(c) > 40]  # discard tiny fragments


def pdf_to_chunks(path: Path) -> tuple[PaperMeta, list[Chunk]]:
    meta = extract_metadata(path)
    pages = extract_text_pdfplumber(path)
    full = clean_text("\n\n".join(pages))

    raw_chunks = chunk_text(full)
    chunks = [
        Chunk(text=text, chunk_idx=i, meta=meta) for i, text in enumerate(raw_chunks)
    ]
    return meta, chunks


# ── Embedding ─────────────────────────────────────────────────────────────────


class BGEEmbedder:
    """Wraps BAAI/bge-small-en with the correct query/passage prefixes."""

    def __init__(self, model_name: str = EMBED_MODEL):
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        log.info("Model loaded (dim=%d)", self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        vec = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        )
        return vec.tolist()


# ── Vector DB ─────────────────────────────────────────────────────────────────


def get_collection(db_dir: Path = DB_DIR, collection: str = COLLECTION):
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for BGE
    )
    return client, col


def already_ingested(col, file_hash: str) -> bool:
    """Check if any chunk from this file already exists in the collection."""
    res = col.get(where={"file_hash": {"$eq": file_hash}}, limit=1)
    return len(res["ids"]) > 0


# ── Main ingestion pipeline ───────────────────────────────────────────────────


def ingest_pdfs(
    pdf_dir: Path = PDF_DIR,
    db_dir: Path = DB_DIR,
    collection: str = COLLECTION,
    batch_size: int = 64,
    force: bool = False,
) -> dict:
    """
    Main entry point.  Returns a summary dict.

    Parameters
    ----------
    pdf_dir    : folder containing PDFs
    db_dir     : where ChromaDB is persisted
    collection : ChromaDB collection name
    batch_size : embedding batch size
    force      : re-ingest even if file hash already present
    """
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True)
        log.warning("Created %s — add your PDFs there, then re-run.", pdf_dir)
        return {"status": "empty_dir", "ingested": 0, "skipped": 0, "errors": []}

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", pdf_dir)
        return {"status": "no_pdfs", "ingested": 0, "skipped": 0, "errors": []}

    log.info("Found %d PDF(s) in %s", len(pdf_files), pdf_dir)

    embedder = BGEEmbedder()
    _, col = get_collection(db_dir, collection)
    ingested, skipped = 0, 0
    errors: list[str] = []

    for pdf_path in tqdm(pdf_files, desc="PDFs", unit="file"):
        try:
            log.info("Processing: %s", pdf_path.name)
            meta, chunks = pdf_to_chunks(pdf_path)

            if not force and already_ingested(col, meta.file_hash):
                log.info("  ↳ already in DB — skipping (use force=True to re-ingest)")
                skipped += 1
                continue

            if not chunks:
                log.warning("  ↳ no text extracted — skipping")
                errors.append(f"{pdf_path.name}: no text extracted")
                continue

            # Embed in batches
            all_texts = [c.text for c in chunks]
            all_ids = [c.chunk_id for c in chunks]
            all_metas = [
                {
                    "filename": c.meta.filename,
                    "title": c.meta.title or c.meta.filename,
                    "authors": c.meta.authors,
                    "year": c.meta.year,
                    "abstract": c.meta.abstract[:500],  # Chroma meta has a size limit
                    "doi": c.meta.doi,
                    "pages": c.meta.pages,
                    "file_hash": c.meta.file_hash,
                    "chunk_idx": c.chunk_idx,
                }
                for c in chunks
            ]

            all_vecs: list[list[float]] = []
            for i in range(0, len(all_texts), batch_size):
                batch_vecs = embedder.embed_passages(all_texts[i : i + batch_size])
                all_vecs.extend(batch_vecs)

            # Upsert into ChromaDB
            col.upsert(
                ids=all_ids,
                embeddings=all_vecs,
                documents=all_texts,
                metadatas=all_metas,
            )

            log.info(
                "  ↳ %d chunks ingested  |  title: %s", len(chunks), meta.title or "—"
            )
            ingested += 1

        except Exception as exc:
            log.error("  ↳ ERROR processing %s: %s", pdf_path.name, exc, exc_info=True)
            errors.append(f"{pdf_path.name}: {exc}")

    summary = {
        "status": "done",
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
        "db_path": str(db_dir.resolve()),
        "collection": collection,
        "total_chunks": col.count(),
    }
    log.info("─" * 60)
    log.info(
        "Done.  Ingested=%d  Skipped=%d  Errors=%d", ingested, skipped, len(errors)
    )
    log.info("Vector DB: %s  |  Total chunks: %d", db_dir, col.count())
    if errors:
        log.warning("Errors:\n  " + "\n  ".join(errors))

    # Persist summary as JSON next to the DB
    summary_path = db_dir / "ingestion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Query helper (for testing / downstream use) ───────────────────────────────


def query_db(
    query: str,
    n_results: int = 10,
    db_dir: Path = DB_DIR,
    collection: str = COLLECTION,
) -> list[dict]:
    """
    Semantic search over the vector DB.
    Returns list of {text, score, metadata} dicts sorted by relevance.

    Example
    -------
    results = query_db("transformer attention mechanism survey")
    for r in results:
        print(r["metadata"]["title"], "—", r["score"])
        print(r["text"][:200])
    """
    embedder = BGEEmbedder()
    _, col = get_collection(db_dir, collection)
    query_vec = embedder.embed_query(query)

    res = col.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "score": round(1 - dist, 4),  # cosine similarity
                "metadata": meta,
            }
        )
    return hits


# ── Heuristic helpers (keep at bottom) ───────────────────────────────────────


def _clean(s) -> str:
    return str(s).strip() if s else ""


def _heuristic_title(text: str, fallback: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Title is usually the longest line in the first 10 lines with ≥4 words
    candidates = [l for l in lines[:10] if 4 <= len(l.split()) <= 25]
    return candidates[0] if candidates else fallback


def _heuristic_authors(text: str) -> str:
    # Look for "Author(s):" label or a line of comma-separated proper nouns
    m = re.search(r"(?:authors?|by)[:\s]+([A-Z][^\n]{5,80})", text, re.I)
    if m:
        return m.group(1).strip()
    return ""


def _heuristic_year(text: str) -> str:
    m = re.search(r"\b(19[89]\d|20[0-2]\d)\b", text)
    return m.group(1) if m else ""


def _heuristic_doi(text: str) -> str:
    m = re.search(r"10\.\d{4,9}/[^\s\"<>]+", text)
    return m.group(0) if m else ""


def _heuristic_abstract(text: str) -> str:
    m = re.search(
        r"(?:abstract|summary)[:\s]*\n?(.*?)(?:\n{2,}|introduction|keywords)",
        text,
        re.I | re.S,
    )
    if m:
        return clean_text(m.group(1))[:1000]
    return ""


def ingest_pdf_from_user():
    PDF_DIR = Path("./pdf-from-user")
    DB_DIR = Path("../chroma_db")
    COLLECTION = "literature_db"

    BATCH_SIZE = 64
    FORCE_REINGEST = False
    TEST_QUERY = None

    summary = ingest_pdfs(
        pdf_dir=PDF_DIR,
        db_dir=DB_DIR,
        collection=COLLECTION,
        batch_size=BATCH_SIZE,
        force=FORCE_REINGEST,
    )
