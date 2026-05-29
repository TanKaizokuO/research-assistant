# Research Assistant - Backend Current State

The backend is built as a **FastAPI** application that serves as the core orchestration layer for the AI Research Assistant. It interfaces with academic APIs, a local vector database, web search APIs, and NVIDIA hosted LLMs (meta/llama) via LangChain and LangGraph.

---

## 🛠️ Technology Stack
- **Framework**: FastAPI
- **LLM Integration**: LangChain (`langchain_nvidia_ai_endpoints`)
- **Agent Orchestration**: LangGraph (StateGraph)
- **Vector Database**: ChromaDB
- **Academic APIs**: Semantic Scholar, arXiv API (via `langchain_community.utilities.arxiv`)
- **Web Search**: Tavily Search API

---

## 📂 Code Structure & Components

### 1. API Routing Layer (`backend/api/`)
- **`app.py`**: Global application factory, configures CORS middleware, registers rate limiters, and binds the routers.
- **`routers/`**:
  - `research.py` (`/research/`): Endpoint for direct multi-source topic research.
  - `literature.py` (`/literature/`): Handles user PDF uploads and initiates literature reviews.
  - `citations.py` (`/citations/`): Entrypoint to retrieve a paper's citation landscape.
- **`agent/router.py`** (`/agent/`): Invokes the ReAct agent using **Server-Sent Events (SSE)**. Streams output chunks (`answer_chunk`), tool executions (`tool`), metadata (`session_id`), and errors to the client.

### 2. LangGraph ReAct Agent (`backend/api/agent/`)
- **`agent.py`**: Builds the agent workflow graph:
  - **Router Node**: Uses `meta/llama-3.1-8b-instruct` with structured output to dynamically select which tools are required for the user's query.
  - **Agent Node**: Primary reasoning model using `meta/llama-3.3-70b-instruct`. 
  - **Tool Loop & Constraints**: Implements a hard cap of **3 tool rounds** to prevent run-away loops.
- **`tools.py`**: Implements 4 LangChain tools:
  - `research_topic`: Synthesizes arXiv, Semantic Scholar, and Web search.
  - `literature_review`: Queries the local ChromaDB for uploaded PDFs.
  - `citation_graph`: Performs citation network lookup.
  - `ingest_pdf`: Prevents direct file streaming over text agent; prompts UI upload.
- **`memory.py`**: Local conversation buffer memory to preserve session context.

### 3. Data Loaders (`backend/data_loaders/`)
- **`semantic_scholar_loader.py`**: Interfaces with the Semantic Scholar API. Handles rate-limiting (HTTP 429), fetches paper records, references, and citations, resolving full metadata (authors, venue, publication year, PDF links).
- **`arxiv_loader.py`**: Uses `ArxivAPIWrapper` to search, rank via a summarizer LLM, download, and write selected papers into local Markdown files inside the `arxiv/` folder.
- **`pdf_ingestion.py`**: Parses local PDFs, splits the text into chunks, and computes/persists embeddings to ChromaDB.
- **`web_search.py`**: Performs Tavily API search.

### 4. Orchestration Services (`backend/services/`)
- **`research_service.py`**: Runs web, arXiv, and Semantic Scholar queries concurrently, builds a consolidated context snippet, and prompts the LLM to write a research brief.
- **`literature_service.py`**: Handles Vector DB querying. Fallbacks to arXiv and Semantic Scholar if local hits are below a threshold (`min_hits`), generating structured academic reviews.
- **`citation_service.py`**: Performs paper ID resolution, concurrently fetches references and citing papers, and generates a landscape summary.

---

## ⚙️ Configuration & Environment Variables
Configured via the `backend/.env` file:
- `NVIDIA_API_KEY`: NVIDIA AI Foundation endpoints authentication.
- `TAVILY_API_KEY`: Web search API.
- `SEMANTIC_SCHOLAR_API_KEY`: For higher rate-limits on Semantic Scholar graph lookups.
- `LANGSMITH_*`: LangSmith tracing project setup for debug logs.

---

## 🏃 Running the Backend
To start the development server:
```bash
cd backend
../.venv/bin/python main.py
```
This runs the application on `http://localhost:8000` with hot-reloading enabled.
