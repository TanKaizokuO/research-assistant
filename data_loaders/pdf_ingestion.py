import os
from langchain_community.document_loaders import PyPDFLoader
from output_schemas.schema import DataSchema
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import requests
from logger import get_logger
from langchain_core.messages import HumanMessage


logger = get_logger(__name__)
SUMMARISER_MODEL = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")


# assuming SourceSchema is already defined


def generate_source_id(file_path: str) -> str:
    """Generate a deterministic source identifier from a PDF file path.

    This utility creates a stable MD5 hash from the provided file path so each
    ingested document can be tracked with a repeatable ID across runs. It is
    used when constructing structured source objects for downstream retrieval,
    ranking, and attribution.

    Attributes:
        file_path (str): Path to the PDF file that should be uniquely identified.

    Example:
        source_id = generate_source_id("pdf-from-user/report.pdf")
    """
    return hashlib.md5(file_path.encode()).hexdigest()


def extract_summary(text: str, max_len: int = 1000) -> str:
    """Produce a concise LLM-generated summary from raw PDF text.

    This function normalizes input content and asks the summarization model to
    compress it into a shorter, readable form bounded by a target character
    length. The returned text is intended for quick previews and metadata fields
    in the structured ingestion output.

    Attributes:
        text (str): Raw extracted document text to summarize.
        max_len (int): Maximum desired length for the returned summary.

    Example:
        summary = extract_summary(long_document_text, max_len=600)
    """
    text = text.strip().replace("\n", " ")
    prompt = f"Summarise the following text into less than {max_len} characters: {text}"
    truncated = SUMMARISER_MODEL.invoke(HumanMessage(content=prompt))

    return truncated.content.strip()


def ingest_from_user_uploaded_pdfs(
    query: str, PDF_DIR: str = "pdf-from-user", url: str = None
) -> list[DataSchema]:
    """Ingest PDF files from a directory and convert them into DataSchema items.

    This ingestion pipeline scans a directory for PDFs, extracts text content,
    computes a model-based confidence score against the research query, and
    builds normalized source records for agent consumption.

    Attributes:
        query (str): User research query used to score content relevance.
        PDF_DIR (str): Directory containing local PDF files to ingest.
        url (str): Optional origin URL to attach to each generated source.

    Example:
        sources = ingest_from_user_uploaded_pdfs(
            query="What are current advances in retrieval-augmented generation?",
            PDF_DIR="pdf-from-user"
        )
    """

    sources = []

    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(PDF_DIR, filename)

        try:
            loader = PyPDFLoader(file_path=file_path, mode="single", pages_delimiter="")

            documents = loader.load()

            if not documents:
                continue

            # Since mode="single", take the first document
            doc = documents[0]
            text = doc.page_content
            logger.info("Extracted text from %s (length=%d)", filename, len(text))

            if not text or len(text.strip()) < 50:
                continue  # skip bad parses

            prompt = (
                f"Assess the quality of the following content according to the query: {query}.\n\n"
                "Return ONLY a single integer between 0 and 100.\n\n"
                "STRICT RULES:\n"
                "- Output must be ONLY a number\n"
                "- No words, no explanation, no symbols\n"
                "- No '%' sign\n"
                "- No newlines before or after\n\n"
                "Valid outputs:\n"
                "23\n78\n100\n\n"
                "Invalid outputs:\n"
                "Score: 78\n78%\nThe answer is 78\n\n"
                f"Content:\n{text}"
            )
            confidence = SUMMARISER_MODEL.invoke(HumanMessage(content=prompt))

            source = DataSchema(
                source_id=generate_source_id(file_path),
                title=filename.replace(".pdf", "").strip(),
                source_type="pdf",
                retrieval_timestamp=datetime.now(timezone.utc),
                summary=extract_summary(text),
                content=text,
                url=url,  # optional URL if this PDF was downloaded from the web
                confidence_score=int(confidence.content),
            )

            sources.append(source)

        except Exception as e:
            logger.info(f"Failed to load {file_path}: {e}")

    logger.info("Created %d structured source records from PDFs", len(sources))
    return sources


def download_pdf_from_url(url: str, save_dir: str = "pdf-from-url") -> str:
    """Download a PDF from a URL and persist it to local storage.

    This helper fetches a remote resource with streaming, validates that the
    response appears to be a PDF, and writes the binary content to disk in the
    target directory. It returns the saved file path for later ingestion or
    returns None when the download fails.

    Attributes:
        url (str): HTTP or HTTPS URL pointing to a PDF resource.
        save_dir (str): Local directory where the downloaded file is stored.

    Example:
        saved_path = download_pdf_from_url(
            "https://example.org/paper.pdf",
            save_dir="pdf-from-url"
        )
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract filename from URL or fallback
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or "downloaded.pdf"

    if not filename.endswith(".pdf"):
        filename += ".pdf"

    file_path = os.path.join(save_dir, filename)

    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()

        # Validate content type (important)
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            raise ValueError(
                f"URL does not appear to be a PDF (Content-Type: {content_type})"
            )

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info("Successfully downloaded PDF from %s to %s", url, file_path)
        return file_path

    except Exception as e:
        logger.info("Failed to download PDF from %s: %s", url, e)
        return None
