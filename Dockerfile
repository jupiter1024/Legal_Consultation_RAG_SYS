# ═══════════════════════════════════════════════════════════════════════════
#  LinuxGPT — Dockerfile
#  Multi-stage friendly, production-ready Python 3.11 slim image
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Metadata
LABEL maintainer="LinuxGPT Team"
LABEL description="Linux Documentation RAG API with LLM Factory Pattern"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed by faiss-cpu and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .env.example ./.env.example

# Create data directories (will be mounted as volumes in production)
RUN mkdir -p Data/PDFS Data/Processed_Chunks Data/VectorStore

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
