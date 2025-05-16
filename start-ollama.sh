#!/bin/sh

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
sleep 5

# Pull and run <YOUR_MODEL_NAME>
ollama pull hf.co/phureexd/qwen3_v2_gguf:Q4_K_M
