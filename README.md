# Research Assistant FastAPI Backend

An AI-powered FastAPI backend providing intelligent research capabilities, literature reviews, and citation graph analysis. It integrates with Tavily for web searching, arXiv, Semantic Scholar, and uses NVIDIA LLMs via LangChain for reasoning and summarization.

## Features

1. **Topic Research (`/research/`)**: Fans out queries concurrently across the Web (Tavily), arXiv, and Semantic Scholar, then produces a comprehensive LLM-generated research brief based on the combined context.
2. **Literature Review (`/literature/`)**:
   - **Ingest**: Upload and ingest user-provided PDFs into a local ChromaDB vector store (`/literature/ingest`).
   - **Review**: Generate structured academic literature reviews by querying the vector DB, optionally supplementing with arXiv and Semantic Scholar if more context is needed (`/literature/review`).
3. **Citation & Reference Finder (`/citations/`)**: Resolves papers on Semantic Scholar to fetch both references (what the paper builds on) and citations (its impact), then generates a citation-landscape summary.

## Setup & Installation

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment variables. Create a `.env` file in the project root with the following keys:
   ```env
   TAVILY_API_KEY=tvly-...
   NVIDIA_API_KEY=nvapi-...
   # SEMANTIC_SCHOLAR_API_KEY=... (Optional: increases rate limits if provided)
   ```

## Running the Server

Start the application using `uvicorn` (recommended for development):
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Alternatively, run it via the main script:
```bash
python main.py
```

## API Documentation

Once the server is running, interactive API documentation is automatically generated and available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Key Endpoints

- `POST /research/` - Search and summarize a research topic.
- `POST /literature/ingest` - Upload `.pdf` files for background ingestion.
- `POST /literature/review` - Generate an academic literature review from ingested documents.
- `POST /citations/` - Explore the citation and reference graph of a specific paper.
- `GET /health` - API health check endpoint.
