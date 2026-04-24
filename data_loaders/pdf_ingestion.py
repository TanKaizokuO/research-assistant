import os
from langchain_community.document_loaders import PyPDFLoader
from output_schemas.schema import DataSchema
import hashlib
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import requests
from logger import (
    get_logger,
    log_tool_call,
    log_tool_result,
    log_agent_action,
    log_warning,
    log_debug,
)


logger = get_logger(__name__)
SUMMARISER_MODEL = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")


# assuming SourceSchema is already defined


def generate_source_id(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()


def extract_summary(text: str, max_len: int = 1000) -> str:
    text = text.strip().replace("\n", " ")
    truncated = SUMMARISER_MODEL.invoke(
        f"Summarise the following text into less than {max_len} characters: {text}"
    )

    return truncated.content.strip()


def ingest_from_user_uploaded_pdfs(
    query: str, PDF_DIR: str = "pdf-from-user", url: str = None
) -> list[DataSchema]:

    sources = []

    logger.info("Starting PDF ingestion from directory: %s", PDF_DIR)

    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(PDF_DIR, filename)
        logger.info("Processing PDF file: %s", file_path)

        try:
            loader = PyPDFLoader(file_path=file_path, mode="single", pages_delimiter="")

            documents = loader.load()
            logger.info("Loaded %d document chunk(s) from %s", len(documents), filename)

            if not documents:
                logger.info("Skipping %s because no documents were extracted", filename)
                continue

            # Since mode="single", take the first document
            doc = documents[0]
            text = doc.page_content

            if not text or len(text.strip()) < 50:
                logger.info(
                    "Skipping %s because extracted content is too short or empty",
                    filename,
                )
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
            confidence = SUMMARISER_MODEL.invoke(prompt)

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
            logger.info("Created source record for %s", filename)

        except Exception as e:
            logger.info(f"Failed to load {file_path}: {e}")

        logger.info("Created %d structured source records from PDFs", len(sources))
    return sources


def download_pdf_from_url(url: str, save_dir: str = "pdf-from-url") -> str:
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

        return file_path

    except Exception as e:
        print(f"Failed to download PDF from {url}: {e}")
        return None
