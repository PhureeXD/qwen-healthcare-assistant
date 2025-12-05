    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir -r requirements.txt 

# Set HF_HOME to ensure models are stored in a consistent location
ENV HF_HOME=/app/hf_cache

# Pre-download models
COPY download_models.py .
RUN python download_models.py 

# Copy application files
COPY . . 
COPY chroma_db /app/chroma_db

# Fix line endings and permissions for shell scripts
RUN dos2unix /app/start-ollama.sh && chmod +x /app/start-ollama.sh

# Expose FastAPI port
EXPOSE 7860

# Start Ollama and FastAPI
CMD ["/bin/sh", "-c", "/app/start-ollama.sh && uvicorn main_v2:app --host 0.0.0.0 --port 7860"]