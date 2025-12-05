#!/bin/sh

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
sleep 5

# Create a custom model using the Modelfile
ollama create custom-model -f /app/Modelfile.local

