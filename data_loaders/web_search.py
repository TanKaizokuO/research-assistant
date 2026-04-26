from calendar import c
from re import S

from logger import (
    get_logger,
    log_tool_call,
    log_tool_result,
    log_agent_action,
    log_warning,
    log_debug,
)
from tavily import TavilyClient
from datetime import datetime, timezone
from output_schemas.schema import SourceSchema
from typing import List
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.tools import tool
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

# Free tier extract limit: up to 20 URLs per call, but keep batches small to
# avoid hitting rate limits or partial-failure blast radius.
_EXTRACT_BATCH_SIZE = 5
_EXTRACT_MAX_URLS = 20  # hard cap to stay within free-tier usage
_SUMMARY_MAX_LENGTH = 2000  # must match SourceSchema.summary max_length
SUMMARISER_MODEL = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")


def manual_web_search(
    tavily_api_key: str,
    query: str,
    max_results: int = 5,
    nvidia_api_key: str = None,
) -> List[SourceSchema]:
    """Search the web with Tavily and return normalized source records.

    This function performs a ranked search for the given query, extracts
    content from returned URLs in small batches, and converts the results into
    `SourceSchema` objects suitable for downstream agent reasoning. When full
    extraction fails for a URL, it falls back to Tavily snippets to preserve
    coverage while maintaining stable output structure.

    Attributes:
        tavily_api_key (str): Tavily API key used for search and extraction.
        query (str): User query used to retrieve relevant web sources.
        max_results (int): Maximum number of ranked search results requested.
        nvidia_api_key (str): Optional NVIDIA API key placeholder for tool
            compatibility; currently not required by this function.

    Example:
        sources = manual_web_search(
            tavily_api_key="tvly-...",
            query="latest retrieval augmented generation methods",
            max_results=5,
        )
    """
    tavily = TavilyClient(api_key=tavily_api_key)

    # ------------------------------------------------------------------ #
    # 1. Search                                                            #
    # ------------------------------------------------------------------ #
    try:
        search_response = tavily.search(query=query, max_results=max_results)
    except Exception:
        logger.error(f"Search failed for query: {query!r}", exc_info=True)
        return []

    raw_results: list[dict] = search_response.get("results", [])
    if not raw_results:
        logger.warning("Search returned no results.")
        return []

    # Build an ordered, deduplicated URL list while keeping per-URL metadata
    # (snippet, title) so we can fall back to it if extraction fails.
    seen: set[str] = set()
    ordered_meta: list[dict] = []  # preserves search rank
    for r in raw_results:
        url = r.get("url", "").strip()
        if url and url not in seen and not url.lower().endswith(".pdf"):
            seen.add(url)
            ordered_meta.append(
                {
                    "url": url,
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),  # Tavily snippet field
                }
            )

    urls_to_extract = [m["url"] for m in ordered_meta][:_EXTRACT_MAX_URLS]
    logger.info(f"URLs to extract ({len(urls_to_extract)}): {urls_to_extract}")

    # ------------------------------------------------------------------ #
    # 2. Extract (batched)                                                 #
    # ------------------------------------------------------------------ #
    extracted: dict[str, str] = {}  # url -> raw_content

    for batch in _chunk(urls_to_extract, _EXTRACT_BATCH_SIZE):
        try:
            res = tavily.extract(urls=batch)
            for item in res.get("results", []):
                url = item.get("url", "")
                content = item.get("raw_content", "").strip()
                if url and content:
                    extracted[url] = content
            # Log any URLs Tavily explicitly failed on
            for failed in res.get("failed_results", []):
                logger.warning(f"Extraction failed for: {failed.get('url')}")
        except Exception:
            logger.error(f"Batch extraction failed: {batch}", exc_info=True)

    logger.info(f"Successfully extracted content for {len(extracted)} URL(s).")

    # ------------------------------------------------------------------ #
    # 3. Build SourceSchema objects                                        #
    # ------------------------------------------------------------------ #
    sources: List[SourceSchema] = []
    for idx, meta in enumerate(ordered_meta):
        url = meta["url"]
        # Prefer full extracted content; fall back to the search snippet.
        content = extracted.get(url) or meta["snippet"]
        if not content:
            logger.debug(f"Skipping {url!r} — no content available.")
            continue
        try:
            prompt = (
                f"Assess the quality of the following content according to the given query: {query}. "
                "Consider factors like relevance, completeness, and readability. "
                "Respond with a single confidence score from 0 to 100, where 100 means "
                "the content is highly relevant, complete, and well-written, and 0 means it is not useful at all."
                f"\n\nContent:\n{content}"
            )
            confidence = SUMMARISER_MODEL.invoke(HumanMessage(content=prompt))
            confidence_score = (
                int(confidence.content.strip()) if url not in extracted else 75
            )
            sources.append(
                _to_source_schema(
                    url=url,
                    title=meta["title"],
                    content=content,
                    idx=idx,
                    confidence_score=confidence_score,
                    used_fallback=url not in extracted,
                )
            )
        except Exception:
            logger.error(f"Failed to build SourceSchema for {url!r}", exc_info=True)

    logger.info(f"Returning {len(sources)} source(s) for query: {query!r}")
    return sources


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _chunk(lst: list, size: int):
    """Yield fixed-size chunks from a list for batched processing.

    This helper splits an input list into consecutive slices of a target size,
    which is useful for sending requests in safe batches and avoiding API
    limits or large failure surfaces in a single call.

    Attributes:
        lst (list): Input sequence to split into chunks.
        size (int): Number of items per yielded chunk.

    Example:
        for batch in _chunk([1, 2, 3, 4, 5], 2):
            print(batch)
    """
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _truncate_to_sentence(text: str, max_len: int) -> str:
    """Reduce text length to fit schema limits while preserving readability.

    This function shortens content to a maximum character limit, preferring a
    sentence boundary when possible so output remains coherent. For oversized
    text, it uses iterative model-based summarization and falls back to a hard
    cutoff strategy if repeated summarization attempts do not converge.

    Attributes:
        text (str): Raw content that may exceed the allowed summary length.
        max_len (int): Maximum output length permitted for the summary field.

    Example:
        summary = _truncate_to_sentence(article_text, 2000)
    """
    if len(text) <= max_len:
        return text

    else:
        c = 0
        while len(text) > max_len:
            truncated = SUMMARISER_MODEL.invoke(
                HumanMessage(
                    content=f"Summarise the following text into less than {max_len} characters: {text}"
                )
            )
            max_len = len(
                truncated.content
            )  # update max_len to the new truncated length
            text = truncated.content
            c += 1
            if c > 3:
                break
    if c > 3:
        truncated = text[:max_len]
        last_period = truncated.rfind(".")
        # Only cut at a sentence boundary if it leaves a reasonable amount of text.
        if last_period > max_len // 2:
            return truncated[: last_period + 1]
        return truncated
    else:
        return text


def _to_source_schema(
    url: str,
    title: str,
    content: str,
    idx: int,
    used_fallback: bool = False,
    confidence_score: int = 70,
) -> SourceSchema:
    """Convert extracted web fields into a validated `SourceSchema` object.

    This helper applies summary truncation, sets stable metadata fields, and
    returns a normalized schema instance used by the research pipeline. It
    centralizes source construction so all web results share consistent typing
    and timestamp behavior.

    Attributes:
        url (str): Canonical URL of the source page.
        title (str): Human-readable page title from search metadata.
        content (str): Extracted or fallback snippet content to summarize.
        idx (int): Ranking index used to build a deterministic source ID.
        used_fallback (bool): Whether snippet fallback was used instead of full
            extraction.
        confidence_score (int): Confidence score assigned to the source.

    Example:
        source = _to_source_schema(
            url="https://example.com/article",
            title="Research Article",
            content="Long extracted content...",
            idx=0,
            confidence_score=82,
        )
    """
    # Confidence is slightly lower when we only have the search snippet.
    # confidence = 70 if used_fallback else 85

    # FIX 1: Truncate before passing to SourceSchema — raw extracted content
    # routinely exceeds max_length=2000, which would raise a ValidationError.
    summary = _truncate_to_sentence(content, _SUMMARY_MAX_LENGTH)

    return SourceSchema(
        source_id=f"src_{idx}",
        url=url,
        title=title,
        source_type="web_article",
        # FIX 2: datetime.utcnow() returns a naive datetime. The schema's
        # future-timestamp validator compares against now(tz=utc), which raises
        # a TypeError when mixed with a naive datetime. Use aware UTC instead.
        retrieval_timestamp=datetime.now(timezone.utc),
        summary=summary,
        confidence_score=confidence_score,
    )
