# AI Research Assistant

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About the Project

**AI Research Assistant** is an AI-powered application designed to streamline the academic research process. It features an interactive, real-time streaming chat interface driven by an agentic reasoning engine. Whether you are conducting a literature review, diving deep into citation networks, or seeking synthesized academic briefs from multiple sources, this tool accelerates your research workflows and helps you uncover meaningful insights with ease.

## Features

- **Interactive ReAct Agent Chat:** Engage in conversational, session-aware interactions where AI dynamically invokes tools (e.g., web search, literature review) right before your eyes.
- **Comprehensive Topic Research:** Synthesize academic briefs seamlessly by pulling data from multiple concurrent sources, including Semantic Scholar, arXiv, and Tavily Web Search.
- **Intelligent Literature Review:** Ingest, parse, and embed local scientific PDFs. Run semantic search queries over these documents and seamlessly fall back to external APIs when broader context is needed.
- **Citation Network Explorer:** Visualize and trace reference networks to map out the academic impact and context of critical research papers.

## Built With

### Backend
- **Python 3.10+**
- **FastAPI** - High-performance web framework for APIs
- **LangGraph & LangChain** - For agent orchestration (ReAct architecture) and tooling
- **ChromaDB** - Local vector database for PDF embeddings

### Frontend
- **React 19 (TypeScript)** - Modern UI development
- **Vite** - Lightning-fast build tool
- **Tailwind CSS / Vanilla CSS** - Custom dark mode themes
- **Server-Sent Events (SSE)** - Real-time streaming

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- **Node.js**: v18.0.0 or higher
- **npm**: v8.0.0 or higher
- **Python**: v3.10.0 or higher

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/research-assistant.git
   cd research-assistant
   ```

2. **Backend Setup**
   ```bash
   # Navigate to the backend directory
   cd backend

   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Backend Environment Variables**
   Create a `.env` file in the `backend/` directory and configure your API keys:
   ```env
   NVIDIA_API_KEY=your_nvidia_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here # Optional
   ```

4. **Frontend Setup**
   ```bash
   # Navigate to the frontend directory
   cd ../frontend

   # Install Node dependencies
   npm install
   ```

5. **Start the Application**
   - **Backend Server**: run `python main.py` in the `backend/` directory (ensure venv is active). It will start on `http://localhost:8000`.
   - **Frontend App**: run `npm run dev` in the `frontend/` directory. It will start on `http://localhost:5173`.
