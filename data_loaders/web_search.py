from logger import (
    get_logger,
    log_tool_call,
    log_tool_result,
    log_agent_action,
    log_warning,
    log_debug,
)
from tavily import TavilyClient
from datetime import datetime
from output_schemas.schema import SourceSchema
import json

logger = get_logger(__name__)


def Tavily_Client(tavily_api_key, query: str, max_results: int = 5) -> SourceSchema:
    """
    Uses Tavily API to search the web, crawl linked pages, and extract content.
    Returns a list of SourceSchema objects with the extracted information.
    """
    tavily = TavilyClient(api_key=tavily_api_key)

    try:
        search_results = tavily.search(query=query, max_results=max_results)
    except Exception as e:
        logger.error(f"Search failed for query: {query}", exc_info=True)
        return []

    urls = [r["url"] for r in search_results["results"]]

    logger.info(f"Search results: {urls}")

    try:
        crawl_results = tavily.crawl(
            url=urls[0], max_depth=1, limit=10  # start from best result
        )
    except Exception as e:
        logger.error(f"Crawl failed for URL: {urls[0]}", exc_info=True)
        crawl_results = {"results": []}

    all_urls = list(set(urls + [r["url"] for r in crawl_results["results"]]))

    logger.info(f"Crawled URLs: {all_urls}")

    all_urls = [u for u in all_urls if not u.endswith(".pdf")]

    # try:
    #     extract_results = tavily.extract(urls=all_urls)
    # except Exception as e:
    #     logger.error(f"Extract failed for URLs: {all_urls}", exc_info=True)
    #     extract_results = {"results": []}

    def chunk(lst, size=5):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    results = []

    for batch in chunk(all_urls, 5):  # try 2–5 max
        try:
            res = tavily.extract(urls=batch)
            if res and "results" in res:
                results.extend(res["results"])
        except Exception as e:
            logger.error(f"Batch failed: {batch}", exc_info=True)

    documents = results
    logger.info(f"Extracted documents: {len(documents)}")

    map_result = tavily.map(url=urls[0])
    with open("map_result.json", "w") as f:
        json.dump(map_result, f, indent=2)

    logger.info(f"Map result: {map_result}")

    sources = [to_source_schema(doc, i) for i, doc in enumerate(documents)]
    return sources


def to_source_schema(doc, idx):
    return {
        "source_id": f"src_{idx}",
        "url": doc["url"],
        "title": doc.get("title", ""),
        "source_type": "web_article",
        "retrieval_timestamp": datetime.utcnow(),
        "summary": doc["raw_content"],
        "confidence_score": 85,
    }
