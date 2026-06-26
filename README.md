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

---

## Usage

Once the application is running, navigate to `http://localhost:5173` in your web browser. 

*Add code examples, diagrams, or screenshots here to demonstrate how users can leverage the AI Research Assistant effectively.*

![App Screenshot Placeholder](https://via.placeholder.com/800x400?text=App+Screenshot)

* Example Use Case 1: Searching for recent papers on LLM agents.
* Example Use Case 2: Uploading a local PDF to perform a deep-dive literature review.

---

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
