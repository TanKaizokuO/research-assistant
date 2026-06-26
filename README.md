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
