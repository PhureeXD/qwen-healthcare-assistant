# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV ANONYMIZED_TELEMETRY=False
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    curl \
    dos2unix \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r requirements.txt 

# Set HF_HOME to ensure models are stored in a consistent location
ENV HF_HOME=/app/hf_cache

# Pre-download models
COPY download_models.py .
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python download_models.py

# Copy application files
COPY . . 
COPY chroma_db /app/chroma_db

# Fix line endings and permissions for shell scripts
RUN dos2unix /app/start-ollama.sh && chmod +x /app/start-ollama.sh

# Expose FastAPI port
EXPOSE 7860

# Start Ollama and FastAPI
CMD ["/bin/sh", "-c", "/app/start-ollama.sh && uvicorn main_v2:app --host 0.0.0.0 --port 7860"]
