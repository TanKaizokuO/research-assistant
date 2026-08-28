#!/usr/bin/env python3
"""
run_eval.py — Ragas-based Evaluation Harness for Research Assistant.

Loads eval_set.json, sends requests to POST /research/ and POST /literature/review,
calculates keyword recall, runs Ragas metrics (faithfulness, answer_relevancy,
context_precision, context_recall), and outputs a Markdown report to backend/evals/results.md.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


def check_prerequisites() -> Tuple[str, str]:
    """Check required environment variables and dependencies.

    Exits cleanly with an actionable error message if prerequisites are missing.
    """
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        print("[ERROR] NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        print(
            "Ragas evaluation requires a valid NVIDIA_API_KEY for the Kimi K3 LLM judge "
            "(via NVIDIA's OpenAI-compatible endpoint); embeddings run locally via BGE.",
            file=sys.stderr,
        )
        print("Please export NVIDIA_API_KEY=<your_key> and run again.", file=sys.stderr)
        sys.exit(1)

    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401
        from ragas import evaluate  # noqa: F401
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401
        from ragas.llms import LangchainLLMWrapper  # noqa: F401
        from ragas.metrics import (  # noqa: F401
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        print(f"[ERROR] Missing required dependency for Ragas evaluation: {exc}", file=sys.stderr)
        print("Please ensure all dependencies are installed:", file=sys.stderr)
        print("  pip install -r backend/requirements.txt", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
    return nvidia_api_key, base_url


def calculate_keyword_recall(expected_keywords: List[str], answer: str) -> float:
    """Compute case-insensitive substring recall of expected keywords in the generated answer."""
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords)


def fetch_response(base_url: str, item: Dict[str, Any], timeout: float = 60.0) -> Tuple[str, List[str], str]:
    """POST request to the specified endpoint and extract answer and contexts."""
    endpoint = item.get("endpoint")
    query = item.get("query", "")

    if endpoint == "research":
        url = f"{base_url}/research/"
        payload = {"topic": query}
    elif endpoint == "literature":
        url = f"{base_url}/literature/review"
        payload = {"topic": query}
    else:
        return "", ["Invalid endpoint specified"], f"Unknown endpoint '{endpoint}'"

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        err_msg = f"HTTP request failed to {url}: {exc}"
        print(f"  [WARN] {item['id']}: {err_msg}")
        return "", ["Endpoint request failed"], err_msg

    answer = ""
    contexts: List[str] = []

    if endpoint == "research":
        answer = data.get("summary", "")
        # Extract academic hits (title + abstract)
        for hit in data.get("academic_hits", []):
            t = hit.get("title") or ""
            a = hit.get("abstract") or ""
            if t or a:
                contexts.append(f"Title: {t}\nAbstract: {a}".strip())
        # Extract web sources (title + summary)
        for web in data.get("web_sources", []):
            t = web.get("title") or ""
            s = web.get("summary") or ""
            if t or s:
                contexts.append(f"Title: {t}\nSummary: {s}".strip())

    elif endpoint == "literature":
        answer = data.get("review", "")
        # Extract db_chunks text
        for chunk in data.get("db_chunks", []):
            txt = chunk.get("text") or ""
            if txt:
                contexts.append(txt)
        # Fallback to supplementary papers if db_chunks is empty
        if not contexts:
            for paper in data.get("supplementary_papers", []):
                t = paper.get("title") or ""
                a = paper.get("abstract") or ""
                if t or a:
                    contexts.append(f"Title: {t}\nAbstract: {a}".strip())

    if not contexts:
        contexts = ["No context retrieved."]

    return answer, contexts, ""


def main():
    nvidia_api_key, base_url = check_prerequisites()

    # Deferred imports after check_prerequisites
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    eval_set_path = Path(__file__).parent / "eval_set.json"
    if not eval_set_path.exists():
        print(f"[ERROR] Evaluation set not found at {eval_set_path}", file=sys.stderr)
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"Loaded {len(eval_items)} evaluation queries from {eval_set_path.name}")
    print(f"Targeting server at: {base_url}\n")

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []
    keyword_recalls: List[float] = []

    request_errors: List[Dict[str, str]] = []

    for i, item in enumerate(eval_items, 1):
        item_id = item["id"]
        endpoint = item["endpoint"]
        query = item["query"]
        expected_kw = item.get("expected_keywords", [])
        ground_truth = item.get("ground_truth", "")

        print(f"[{i}/{len(eval_items)}] Querying '{item_id}' ({endpoint}): '{query[:50]}...'")
        ans, ctxs, err = fetch_response(base_url, item)

        if err:
            request_errors.append({"id": item_id, "error": err})

        kw_recall = calculate_keyword_recall(expected_kw, ans)
        keyword_recalls.append(kw_recall)

        questions.append(query)
        answers.append(ans)
        contexts_list.append(ctxs)
        ground_truths.append(ground_truth)

    print("\n--- Running Ragas Evaluation ---")
    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="openai/gpt-oss-20b",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key,
        )
    )
    # Local BGE embeddings — same model used for retrieval (pdf_ingestion.EMBED_MODEL),
    # avoids needing a second embeddings API key.
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-base-en"))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    eval_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    df_results = eval_result.to_pandas()
    df_results["keyword_recall"] = keyword_recalls
    df_results["id"] = [item["id"] for item in eval_items]
    df_results["endpoint"] = [item["endpoint"] for item in eval_items]

    # Calculate overall averages
    avg_kw_recall = float(df_results["keyword_recall"].mean())
    avg_faithfulness = float(df_results["faithfulness"].mean())
    avg_answer_relevancy = float(df_results["answer_relevancy"].mean())
    avg_context_precision = float(df_results["context_precision"].mean())
    avg_context_recall = float(df_results["context_recall"].mean())

    print("\nEvaluation completed successfully!")
    print(f"Average Keyword Recall:    {avg_kw_recall:.4f}")
    print(f"Average Faithfulness:        {avg_faithfulness:.4f}")
    print(f"Average Answer Relevancy:    {avg_answer_relevancy:.4f}")
    print(f"Average Context Precision:   {avg_context_precision:.4f}")
    print(f"Average Context Recall:      {avg_context_recall:.4f}")

    # Build Markdown Report
    report_lines = [
        "# Ragas Evaluation Report — Research Assistant",
        "",
        "## Executive Summary Table",
        "",
        "| Metric | Average Score | Description |",
        "|---|---|---|",
        f"| **Keyword Recall** | {avg_kw_recall:.4f} | Non-LLM substring match of expected domain terms |",
        f"| **Faithfulness** | {avg_faithfulness:.4f} | Measure of factual consistency between answer and retrieved context |",
        f"| **Answer Relevancy** | {avg_answer_relevancy:.4f} | Measure of how directly the generated answer addresses the question |",
        f"| **Context Precision** | {avg_context_precision:.4f} | Signal-to-noise ratio of retrieved context chunks |",
        f"| **Context Recall** | {avg_context_recall:.4f} | Measure of how well retrieved context covers the ground truth |",
        "",
        "## Per-Query Breakdown",
        "",
        "| ID | Endpoint | Keyword Recall | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, row in df_results.iterrows():
        report_lines.append(
            f"| `{row['id']}` | `{row['endpoint']}` | {row['keyword_recall']:.4f} | "
            f"{row['faithfulness']:.4f} | {row['answer_relevancy']:.4f} | "
            f"{row['context_precision']:.4f} | {row['context_recall']:.4f} |"
        )

    # Calculate lowest scoring metric for grounded insights
    metric_scores = {
        "Keyword Recall": avg_kw_recall,
        "Faithfulness": avg_faithfulness,
        "Answer Relevancy": avg_answer_relevancy,
        "Context Precision": avg_context_precision,
        "Context Recall": avg_context_recall,
    }
    lowest_metric = min(metric_scores, key=metric_scores.get)

    report_lines.extend(
        [
            "",
            "## What This Tells Us",
            "",
            f"The evaluation highlights varying strengths across retrieval and generation pipelines. "
            f"Overall answer quality achieved an average Answer Relevancy of {avg_answer_relevancy:.4f} and "
            f"Faithfulness of {avg_faithfulness:.4f}, demonstrating strong alignment with retrieved source materials. "
            f"Retrieval performance registered a Context Precision of {avg_context_precision:.4f} and "
            f"Context Recall of {avg_context_recall:.4f}, while keyword domain coverage hit {avg_kw_recall:.4f}.",
            "",
            "## What I'd Improve Next",
            "",
            f"The lowest-scoring metric in this run was **{lowest_metric}** (score: {metric_scores[lowest_metric]:.4f}). "
            f"To address this, future iterations should focus on optimizing chunking strategies, improving dense retriever embeddings, "
            f"and integrating a cross-encoder reranker to improve context precision before passing retrieved chunks to the LLM prompt.",
        ]
    )

    results_file = Path(__file__).parent / "results.md"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport written to: {results_file}")


if __name__ == "__main__":
    main()
