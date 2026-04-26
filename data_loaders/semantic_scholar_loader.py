from httpx import get
import requests
import time
from logger import get_logger
import os
from dotenv import load_dotenv
import threading
from typing import List, Dict, Optional
import json

lock = threading.Lock()
load_dotenv()
logger = get_logger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")}  # optional


def paper_search(query: str, limit: int = 5) -> list:
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,openAccessPdf",
    }

    logger.info("Searching papers for query=%s limit=%s", query, limit)
    response = requests.get(url, params=params, headers=HEADERS)
    time.sleep(1)  # rate limit
    response.raise_for_status()

    data = response.json().get("data", [])
    logger.info("Paper search returned %s results", len(data))

    return [
        {
            "paperId": p["paperId"],
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "year": p.get("year"),
            "citations": p.get("citationCount"),
            "pdf": (p.get("openAccessPdf") or {}).get("url"),
        }
        for p in data
    ]


def snippet_search(query: str, limit: int = 3) -> list:
    url = f"{BASE_URL}/snippet/search"
    params = {"query": query, "limit": limit}

    logger.info("Searching snippets for query=%s limit=%s", query, limit)
    response = requests.get(url, params=params, headers=HEADERS)
    time.sleep(1)  # rate limit
    response.raise_for_status()

    data = response.json().get("data", [])
    logger.info("Snippet search returned %s results", len(data))

    return [
        {
            "text": s.get("snippet", {}).get("text"),
            "paperId": s.get("paper", {}).get("paperId"),
            "title": s.get("paper", {}).get("title"),
        }
        for s in data
    ]


def get_paper_id(paper_id):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/match"

    params = {
        "query": paper_id,
        "limit": 5,
    }

    response = rate_limited_get(url, params=params, headers=HEADERS)
    response_json = response.json()

    return response_json.get("data")[0]["paperId"]


# -------------------------------
# Core HTTP helper (robust)
# -------------------------------
def rate_limited_get(url, params=None, headers=None, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else (2**attempt)

            print(f"[429] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response

    raise Exception("Max retries exceeded")


# -------------------------------
# API wrappers
# -------------------------------
def get_paper(paper_id, fields="title,abstract,year,authors"):
    url = f"{BASE_URL}/paper/{paper_id}"

    params = {}
    if fields:
        params["fields"] = fields

    response = rate_limited_get(url, params=params, headers=HEADERS)
    return response.json()


def get_paper_references(paper_id, fields=None, limit=100, offset=0):
    url = f"{BASE_URL}/paper/{paper_id}/references"

    params = {"limit": limit, "offset": offset}
    if fields:
        params["fields"] = fields

    response = rate_limited_get(url, params=params, headers=HEADERS)
    return response.json()


def get_paper_citations(paper_id, fields=None, limit=100, offset=0):
    url = f"{BASE_URL}/paper/{paper_id}/citations"

    params = {"limit": limit, "offset": offset}
    if fields:
        params["fields"] = fields

    response = rate_limited_get(url, params=params, headers=HEADERS)
    return response.json()
