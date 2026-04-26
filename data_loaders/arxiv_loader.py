import os
import re
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.messages import HumanMessage
from logger import get_logger
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from urllib.parse import urlparse

logger = get_logger(__name__)
SUMMARISER_MODEL = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")


def extract_arxiv_id(url: str) -> str:
    path = urlparse(url).path  # e.g. "/abs/2306.14753v1"
    return path.split("/")[-1]


def fetch_and_save_best_arxiv_paper(query: str):
    """Fetch, rank, and persist the most relevant arXiv papers for a query.

    This function searches arXiv for candidate papers, asks an LLM to select
    the most relevant results based on summary text, downloads the selected
    full papers, and saves them as Markdown files in a local `arxiv` folder.
    It returns both the chosen indices and metadata about the files that were
    successfully written, making it suitable for research-assistant ingestion
    and traceable offline review.

    Attributes:
        query (str): Research question or topic used to search arXiv papers.
        llm: Language model client with an `invoke(...)` method used to rank
            candidate summaries.

    Example:
        result = fetch_and_save_best_arxiv_paper(
            query="retrieval-augmented generation for scientific QA",
            llm=chat_model,
        )
        print(result["saved_papers"])
    """

    # Step 1
    logger.info("Starting arXiv fetch for query: %s", query)
    arxiv = ArxivAPIWrapper(
        top_k_results=5, doc_content_chars_max=5000, load_max_docs=5
    )

    # Step 2
    docs = arxiv.get_summaries_as_docs(query)
    logger.info("Fetched %d arXiv summary document(s)", len(docs))
    if not docs:
        raise ValueError("No documents found")

    summaries_text = "\n\n".join(
        [
            f"[{i}]\nTitle: {d.metadata.get('Title')}\nSummary: {d.page_content}"
            for i, d in enumerate(docs)
        ]
    )

    summaries_list = [
        f"[{i}]\nTitle: {d.metadata.get('Title')}\nSummary: {d.page_content}"
        for i, d in enumerate(docs)
    ]

    # Step 3
    prompt = f"""You are selecting the most relevant research papers.

Query: {query}

Papers:
{summaries_text}

Return ONLY the indices (integers) of the TOP 3 most relevant papers.
Format strictly as: 0,2,4
"""

    response = SUMMARISER_MODEL.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    indices = list(map(int, re.findall(r"\d+", raw)))

    if not indices:
        raise ValueError(f"Invalid LLM output: {raw}")

    indices = [i for i in indices if 0 <= i < len(docs)][:3]

    if not indices:
        raise ValueError(f"No valid indices found in: {raw}")

    logger.info("Selected arXiv paper indices: %s", indices)

    # ✅ FIX 1: define selected_docs properly
    selected_docs = [(i, docs[i]) for i in indices]
    # selected_docs = []

    # for i in indices:
    #     d = docs[i]

    #     selected_summaries = [
    #         f"[{i}]\nTitle: {d.metadata.get('Title')}\nSummary: {d.page_content}"
    #         for i, d in enumerate(docs)
    #     ]
    #     selected_docs = (i, d)

    # Step 4 + 5
    os.makedirs("arxiv", exist_ok=True)
    saved_results = []

    for idx, doc in selected_docs:
        title = doc.metadata.get("Title")
        entry_id = doc.metadata.get("Entry ID")  # ✅ FIX 2 (use this!)
        arxiv_id = extract_arxiv_id(entry_id)  # ✅ FIX 3 (extract the ID)

        try:
            full_docs = arxiv.load(arxiv_id)

            if not full_docs:
                logger.info("Skipping paper with no content: %s", title)

                continue

            full_doc = full_docs[0]
            full_text = full_doc.page_content
            metadata = full_doc.metadata

        except Exception as e:
            logger.info("Failed to load paper %s: %s", title, e)
            continue

        # Save markdown
        safe_title = title.replace(" ", "_").replace("/", "_").replace(":", "_")

        filename = safe_title[:100] + ".md"
        filepath = os.path.join("arxiv", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {metadata.get('Title')}\n\n")
            f.write(f"**Authors:** {metadata.get('Authors')}\n\n")
            f.write(f"**Published:** {metadata.get('Published')}\n\n")
            f.write(f"**Entry ID:** {doc.metadata.get('Entry ID')}\n\n")
            f.write(f"**Summary:** {metadata.get('Summary')}\n\n")
            f.write("---\n\n")
            f.write(full_text)

        saved_results.append(
            {
                "index": idx,
                "title": metadata.get("Title"),
                "file_path": filepath,
            }
        )

    logger.info("Saved %d arXiv paper(s) to local markdown", len(saved_results))

    return {"selected_indices": indices, "saved_papers": saved_results}
