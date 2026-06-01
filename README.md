# AI Research Assistant

An AI-powered research assistant application featuring an interactive, real-time streaming chat interface and an agentic reasoning engine. The project is organized as a monorepo containing a FastAPI backend and a React 19 (TypeScript) frontend.

---

## 📂 Project Structure

```text
Research_Assistant/
├── backend/            # FastAPI Application & AI Agent Logic
│   ├── api/            # API routes and Agent definitions
│   ├── data_loaders/   # Semantic Scholar, arXiv, and PDF parsing loaders
│   ├── services/       # Literature review, citation and research service orchestrations
│   └── requirements.txt
├── frontend/           # React 19 + Vite + TypeScript Web Application
│   ├── src/            # App components, styles, and assets
│   └── package.json
├── backend.md          # Technical documentation for the backend
├── frontend.md         # Technical documentation for the frontend
└── README.md           # General project documentation and setup guide
```

---

## 🛠️ Technology Stack

### Backend
* **Framework**: FastAPI
* **Agent Orchestration**: LangGraph (`StateGraph`) with a ReAct architecture
* **LLM Integration**: LangChain (`langchain_nvidia_ai_endpoints`) using `meta/llama-3.1-8b-instruct` (routing) and `meta/llama-3.3-70b-instruct` (reasoning)
* **Vector Database**: ChromaDB (for local PDF embedding storage and search)
* **Academic APIs**: Semantic Scholar, arXiv (via `langchain_community.utilities.arxiv`)
* **Web Search**: Tavily Search API

### Frontend
* **Framework**: React 19 (TypeScript)
* **Build Tool**: Vite
* **Styling**: Vanilla CSS with custom dark mode theme
* **Markdown Rendering**: `react-markdown`
* **Icons**: `lucide-react`
* **Real-time Communication**: Server-Sent Events (SSE) for streaming agent tokens and tool execution steps

---

## ✨ Features

1. **Interactive ReAct Agent Chat**: Conversational AI interface that preserves session memory and displays tool invocations (`research_topic`, `literature_review`, etc.) in real-time as they run.
2. **Topic Research**: Concurrently queries Tavily (Web Search), arXiv, and Semantic Scholar to synthesize a consolidated academic brief.
3. **Literature Review**:
   * **PDF Ingestion**: Upload and parse scientific PDFs, split text into chunks, and persist embeddings into ChromaDB.
   * **Semantic Search**: Run search queries over local PDFs, falling back to external sources when necessary to draft a literature review.
4. **Citation Network Explorer**: Traces references and citations of research papers to compile a landscape summary of their academic context and impact.

---

## ⚙️ Configuration & Environment Setup

### Backend Environment Variables
Create a `.env` file inside the `backend/` directory with the following keys:

```env
NVIDIA_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...
# Optional:
SEMANTIC_SCHOLAR_API_KEY=... # Increases rate limits for Semantic Scholar
LANGSMITH_API_KEY=...         # For tracing and debugging (if using LangSmith)
LANGSMITH_PROJECT=...
```

---

## 🏃 Running the Application

### 1. Setup & Start the Backend

Make sure you have Python 3.10+ installed.

```bash
# From the project root, set up virtual environment and install backend dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Run the backend server
cd backend
../.venv/bin/python main.py
```

The backend API will run on **`http://localhost:8000`**.

* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Setup & Start the Frontend

Make sure you have Node.js (v18+) and npm installed.

```bash
# Install dependencies
cd frontend
npm install

# Run the frontend in development mode
npm run dev
```

The web application will run on **`http://localhost:5173`**.

---

## 📖 Deep Dives

For detailed component mappings, code structure, and design decisions, please refer to the following:
* [Backend Current State Guide](file:///home/tankaizokuo/Code/Research_Assistant/backend.md)
* [Frontend Current State Guide](file:///home/tankaizokuo/Code/Research_Assistant/frontend.md)
