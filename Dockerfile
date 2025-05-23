FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    python3-dev \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir -r requirements.txt 

# Copy application files
COPY . . 
COPY chroma_db /app/chroma_db

# Fix line endings and permissions for shell scripts
RUN dos2unix /app/start-ollama.sh && chmod +x /app/start-ollama.sh

# Expose FastAPI port
EXPOSE 8000

# Start Ollama and FastAPI
CMD ["/bin/sh", "-c", "/app/start-ollama.sh && uvicorn main_v2:app --host 0.0.0.0 --port 8000"]