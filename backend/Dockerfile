# Multi-stage Dockerfile for FastAPI + LangGraph backend

# --- Builder Stage ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools required for Python packages with C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime Stage ---
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user and persistent volume directory
# Note: ChromaDB persistent directory can be configured via CHROMA_DB_DIR env var (defaults to /data/chroma_db or local /app/chroma_db)
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser && \
    mkdir -p /app /data/chroma_db /app/pdf-from-user /app/chroma_db && \
    chown -R appuser:appuser /app /data/chroma_db

WORKDIR /app

# Copy virtualenv and app code from builder
COPY --from=builder /opt/venv /opt/venv
COPY . /app

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Programmatically start uvicorn as configured in backend/main.py
CMD ["python", "main.py"]
