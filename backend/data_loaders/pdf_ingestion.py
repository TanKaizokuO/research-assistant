import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
import pdfplumber
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR = Path("pdf-from-user")
DB_DIR = Path("vector_db")
COLLECTION = "literature_review"

EMBED_MODEL = "BAAI/bge-base-en"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking — tuned for academic papers
CHUNK_SIZE = 512  # tokens (approx chars / 4)
CHUNK_OVERLAP = 80  # overlap keeps context across chunk boundaries

# BGE models need this prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SECTION_HEADINGS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Motivation",
    "Method",
    "Methods",
    "Methodology",
    "Approach",
    "System Design",
    "Experiments",
    "Experimental Setup",
    "Evaluation",
    "Results",
    "Discussion",
    "Analysis",
    "Limitations",
    "Conclusion",
    "Conclusions",
    "Future Work",
    "Acknowledgments",
    "Acknowledgements",
    "References",
    "Bibliography",
    "Appendix",
]

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
    section: str = "Unknown"
    page_start: int = 0
    page_end: int = 0
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


# ── Text cleaning & section-aware chunking ───────────────────────────────────


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


def _detect_section_heading(line: str) -> Optional[str]:
    """
    Heuristically detect canonical academic section headings in a line of text.
    Strips leading numbers/roman numerals and matches against SECTION_HEADINGS.
    """
    s_line = line.strip()
    if not s_line:
        return None

    # Headings rarely end with a period
    if s_line.endswith("."):
        return None

    # Headings are short (<= ~8 words)
    words = s_line.split()
    if len(words) > 8:
        return None

    # Strip leading number/roman-numeral/period pattern (e.g. "3.", "III.", "3.2", "Section 3.")
    cleaned = re.sub(
        r"^(?:(?:Section|Sec\.|Chapter|Chap\.)\s+)?(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+(?:\.[IVXLCDM]+)*|[A-Z]\.)[\.\s]*",
        "",
        s_line,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        cleaned = s_line

    cleaned_lower = cleaned.lower()

    # Exact match check first
    for heading in SECTION_HEADINGS:
        if cleaned_lower == heading.lower():
            return heading

    # Startswith check (sorted longest heading first to avoid false sub-prefix matches)
    sorted_headings = sorted(SECTION_HEADINGS, key=len, reverse=True)
    for heading in sorted_headings:
        h_lower = heading.lower()
        if cleaned_lower.startswith(h_lower):
            match_len = len(h_lower)
            if match_len == len(cleaned_lower) or not cleaned_lower[match_len].isalnum():
                return heading

    return None


def _extract_line_stream(path: Path) -> list[tuple[int, str]]:
    """Build a flat list of (page_num, line) tuples (1-indexed page_num)."""
    pages = extract_text_pdfplumber(path)
    stream: list[tuple[int, str]] = []
    for page_idx, page_text in enumerate(pages, start=1):
        lines = page_text.splitlines()
        for line in lines:
            stream.append((page_idx, line))
    return stream


def _extract_segments(
    line_stream: list[tuple[int, str]]
) -> list[tuple[str, int, int, str]]:
    """
    Group line stream into contiguous section segments:
    (section_name, page_start, page_end, cleaned_text).
    """
    segments: list[tuple[str, int, int, str]] = []
    current_section = "Front Matter"
    current_lines: list[tuple[int, str]] = []

    def flush():
        nonlocal current_lines, current_section
        if not current_lines:
            return
        raw_text = "\n".join(line for _, line in current_lines)
        cleaned_body = clean_text(raw_text)
        if cleaned_body:
            p_start = current_lines[0][0]
            p_end = current_lines[-1][0]
            segments.append((current_section, p_start, p_end, cleaned_body))
        current_lines = []

    for page_num, line in line_stream:
        heading = _detect_section_heading(line)
        if heading is not None:
            flush()
            current_section = heading
        else:
            current_lines.append((page_num, line))

    flush()
    return segments


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
    """
    Page and section aware PDF parsing and chunking pipeline.
    """
    meta = extract_metadata(path)
    line_stream = _extract_line_stream(path)
    segments = _extract_segments(line_stream)

    chunks: list[Chunk] = []
    chunk_idx = 0

    for section_name, seg_p_start, seg_p_end, seg_text in segments:
        seg_len = len(seg_text)
        raw_chunks = chunk_text(seg_text)

        search_pos = 0
        for c in raw_chunks:
            # Approximate per-chunk page range by proportionally mapping character offset
            # within segment text to segment page range (linear interpolation).
            # This is an approximation.
            c_pos = seg_text.find(c, search_pos)
            if c_pos != -1:
                search_pos = c_pos
            else:
                c_pos = seg_text.find(c)
                if c_pos == -1:
                    c_pos = 0

            c_end = c_pos + len(c)

            if seg_len > 0 and seg_p_end > seg_p_start:
                start_ratio = c_pos / seg_len
                end_ratio = c_end / seg_len
                p_start = int(round(seg_p_start + start_ratio * (seg_p_end - seg_p_start)))
                p_end = int(round(seg_p_start + end_ratio * (seg_p_end - seg_p_start)))
                p_start = max(seg_p_start, min(seg_p_end, p_start))
                p_end = max(p_start, min(seg_p_end, p_end))
            else:
                p_start = seg_p_start
                p_end = seg_p_end

            chunk_obj = Chunk(
                text=c,
                chunk_idx=chunk_idx,
                meta=meta,
                section=section_name,
                page_start=p_start,
                page_end=p_end,
            )
            chunks.append(chunk_obj)
            chunk_idx += 1

    return meta, chunks


# ── Embedding & Reranking ─────────────────────────────────────────────────────


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


_EMBEDDER_INSTANCE: Optional["BGEEmbedder"] = None


def _get_embedder() -> "BGEEmbedder":
    """Lazily load and cache the embedding model (avoids reloading it on every call)."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = BGEEmbedder()
    return _EMBEDDER_INSTANCE

_RERANKER_INSTANCE: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    """Lazily load and cache the cross-encoder reranker model."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        log.info("Loading reranker model: %s", RERANK_MODEL)
        _RERANKER_INSTANCE = CrossEncoder(RERANK_MODEL)
        log.info("Reranker model loaded")
    return _RERANKER_INSTANCE


# ── Vector DB & BM25 Index ────────────────────────────────────────────────────

_BM25_CACHE: dict[tuple[str, str], dict] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_bm25_index(db_dir: Path, collection: str, col) -> dict:
    res = col.get(include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    tokenized_corpus = [_tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    entry = {
        "count": len(ids),
        "bm25": bm25,
        "ids": ids,
        "docs": docs,
        "metas": metas,
    }
    cache_key = (str(Path(db_dir).resolve()), str(collection))
    _BM25_CACHE[cache_key] = entry
    return entry


def _get_bm25_index(db_dir: Path, collection: str, col) -> dict:
    cache_key = (str(Path(db_dir).resolve()), str(collection))
    current_count = col.count()
    if cache_key in _BM25_CACHE:
        entry = _BM25_CACHE[cache_key]
        if entry["count"] == current_count:
            return entry
    return _build_bm25_index(db_dir, collection, col)


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
    Main entry point. Returns a summary dict.

    Parameters
    ----------
    pdf_dir : folder containing PDFs
    db_dir : where ChromaDB is persisted
    collection : ChromaDB collection name
    batch_size : embedding batch size
    force : re-ingest even if file hash already present
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

    embedder = _get_embedder()
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
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
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


# ── Query helper ──────────────────────────────────────────────────────────────


def query_db(
    query: str,
    n_results: int = 10,
    db_dir: Path = DB_DIR,
    collection: str = COLLECTION,
) -> list[dict]:
    """
    Hybrid (BM25 + dense embedding) search with cross-encoder reranking over vector DB.
    Returns list of {text, score, metadata} dicts sorted by reranked relevance.
    """
    _, col = get_collection(db_dir, collection)
    total_docs = col.count()

    if total_docs == 0:
        return []

    candidate_k = max(n_results * 3, 20)
    candidate_k = min(candidate_k, total_docs)

    # --- a. Dense pass ---
    embedder = _get_embedder()
    query_vec = embedder.embed_query(query)

    res = col.query(
        query_embeddings=[query_vec],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_ids = res["ids"][0] if res.get("ids") and res["ids"] else []
    dense_docs = res["documents"][0] if res.get("documents") and res["documents"] else []
    dense_metas = res["metadatas"][0] if res.get("metadatas") and res["metadatas"] else []

    doc_map: dict[str, dict] = {}
    for doc_id, doc_text, meta in zip(dense_ids, dense_docs, dense_metas):
        doc_map[doc_id] = {"text": doc_text, "metadata": meta}

    dense_rank_map = {doc_id: rank for rank, doc_id in enumerate(dense_ids)}

    # --- b. BM25 pass ---
    bm25_entry = _get_bm25_index(db_dir, collection, col)
    bm25_ids: list[str] = []
    bm25_rank_map: dict[str, int] = {}

    if bm25_entry["bm25"] is not None and bm25_entry["docs"]:
        tokenized_query = _tokenize(query)
        bm25_scores = bm25_entry["bm25"].get_scores(tokenized_query)
        # Top candidate_k indices by BM25 score
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:candidate_k]

        for rank, idx in enumerate(top_bm25_indices):
            b_id = bm25_entry["ids"][idx]
            b_doc = bm25_entry["docs"][idx]
            b_meta = bm25_entry["metas"][idx]

            bm25_ids.append(b_id)
            bm25_rank_map[b_id] = rank
            if b_id not in doc_map:
                doc_map[b_id] = {"text": b_doc, "metadata": b_meta}

    # --- c. Reciprocal Rank Fusion (RRF) ---
    candidate_ids = list(set(dense_ids) | set(bm25_ids))
    if not candidate_ids:
        return []

    rrf_scores: dict[str, float] = {}
    for cid in candidate_ids:
        score = 0.0
        if cid in dense_rank_map:
            score += 1.0 / (60.0 + dense_rank_map[cid])
        if cid in bm25_rank_map:
            score += 1.0 / (60.0 + bm25_rank_map[cid])
        rrf_scores[cid] = score

    fused_candidates = sorted(
        candidate_ids,
        key=lambda cid: (rrf_scores[cid], cid),
        reverse=True,
    )[:candidate_k]

    if not fused_candidates:
        return []

    # --- d. Cross-encoder rerank pass ---
    pairs = [(query, doc_map[cid]["text"]) for cid in fused_candidates]
    reranker = _get_reranker()
    raw_rerank_scores = reranker.predict(pairs)

    if isinstance(raw_rerank_scores, (float, int)):
        rerank_scores = [float(raw_rerank_scores)]
    else:
        rerank_scores = [float(s) for s in raw_rerank_scores]

    scored_candidates = list(zip(fused_candidates, rerank_scores))
    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    top_results = scored_candidates[:n_results]

    # --- e. Build final output list ---
    hits = []
    for cid, r_score in top_results:
        doc_info = doc_map[cid]
        d_rank = dense_rank_map.get(cid, None)
        b_rank = bm25_rank_map.get(cid, None)
        r_score_val = rrf_scores[cid]

        hit_meta = {
            **doc_info["metadata"],
            "dense_rank": d_rank,
            "bm25_rank": b_rank,
            "rrf_score": round(float(r_score_val), 4),
        }
        hits.append(
            {
                "text": doc_info["text"],
                "score": round(float(r_score), 4),
                "metadata": hit_meta,
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
    DB_DIR = Path("./chroma_db")
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
