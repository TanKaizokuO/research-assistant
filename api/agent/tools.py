"""
api/agent/tools.py — Wraps existing services as LangChain Tools for the ReAct agent.
"""
from langchain_core.tools import tool

from api.dependencies import get_tavily_key
from services.research_service import ResearchRequest, run_research
from services.literature_service import LiteratureRequest, run_literature_review
from services.citation_service import CitationRequest, run_citation_finder


@tool
async def research_topic(topic: str) -> str:
    """
    Use this tool for broad topic overviews, finding recent findings, and general research on a subject.
    Returns a synthesized LLM research brief covering key findings, major themes, and open questions, 
    along with sources from arXiv, Semantic Scholar, and web search.
    
    Args:
        topic (str): The research topic or question to investigate.
    """
    try:
        tavily_key = get_tavily_key()
        req = ResearchRequest(
            topic=topic,
            max_web_results=5,
            max_academic_results=5,
            include_arxiv=True,
            include_semantic_scholar=True,
            include_web=True
        )
        response = await run_research(req, tavily_key)
        return f"Summary:\n{response.summary}\n\nSources examined: {len(response.web_sources) + len(response.academic_hits)} items."
    except Exception as e:
        return f"Error executing research_topic: {str(e)}"


@tool
async def literature_review(topic: str) -> str:
    """
    Use this tool to get a structured literature review based primarily on ingested PDFs in the vector database.
    It will supplement with Semantic Scholar and arXiv if the local database hits are low.
    Use this when the user specifically asks to summarize their uploaded papers or wants a formal literature review.
    
    Args:
        topic (str): The specific research topic or question to structure the review around.
    """
    try:
        tavily_key = get_tavily_key()
        req = LiteratureRequest(
            topic=topic,
            n_db_results=10,
            min_hits=3,
            supplement_with_web=True,
            supplement_with_arxiv=True
        )
        response = await run_literature_review(req, tavily_key)
        return f"Literature Review:\n{response.review}"
    except Exception as e:
        return f"Error executing literature_review: {str(e)}"


@tool
async def citation_graph(query: str) -> str:
    """
    Use this tool to understand a specific paper's impact, finding papers it references and papers that cite it.
    It provides an LLM citation-landscape summary for a given paper title or DOI.
    
    Args:
        query (str): Paper title, DOI, or research topic to look up on Semantic Scholar.
    """
    try:
        req = CitationRequest(
            query=query,
            limit=20,
            include_references=True,
            include_citations=True
        )
        response = await run_citation_finder(req)
        return f"Citation Landscape Summary:\n{response.context_summary}"
    except Exception as e:
        return f"Error executing citation_graph: {str(e)}"


@tool
def ingest_pdf(instruction: str) -> str:
    """
    Use this tool ONLY when the user explicitly asks to upload or ingest a PDF file.
    Since you cannot directly receive files, you will return instructions on how the user can upload PDFs.
    
    Args:
        instruction (str): A dummy string, can be empty.
    """
    return "I cannot process PDF uploads directly via text. Please use the /literature/upload endpoint or the application UI to ingest your PDF files."

