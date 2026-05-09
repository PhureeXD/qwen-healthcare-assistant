# Docker Commands

This app runs `main_v2:app` on port `7860`. The Docker build downloads the embedding model, reranker model, and GGUF model from Hugging Face.

## Build

Use BuildKit and pass your Hugging Face token as a build secret. Do not hardcode the token in the repo or Dockerfile.

```powershell
$env:HF_TOKEN="<your_huggingface_token>"
$env:DOCKER_BUILDKIT="1"
docker build --progress=plain --secret id=hf_token,env=HF_TOKEN -t nlp-app .
```

If you do not need authenticated Hugging Face downloads, you can build without the secret:

```powershell
$env:DOCKER_BUILDKIT="1"
docker build --progress=plain -t nlp-app .
```

## Run

Stop and remove the old container first if it already exists:

```powershell
docker stop nlp-container
docker rm nlp-container
```

Run the new image:

```powershell
docker run --rm -p 7860:7860 --name nlp-container nlp-app
```

Open:

```text
http://localhost:7860/
http://localhost:7860/docs
```

## Test

Clear conversation memory after rebuilding or changing prompts:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:7860/clear"
```

Basic health query:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:7860/generate?query=%E0%B8%AD%E0%B8%A2%E0%B8%B2%E0%B8%81%E0%B8%A3%E0%B8%B9%E0%B9%89%E0%B8%AD%E0%B8%B2%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B9%82%E0%B8%A3%E0%B8%84%E0%B8%AB%E0%B8%B1%E0%B8%A7%E0%B9%83%E0%B8%88"
```

Force RAG manually:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:7860/generate?query=%E0%B8%AD%E0%B8%B2%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%82%E0%B8%AD%E0%B8%87%E0%B9%82%E0%B8%A3%E0%B8%84%E0%B9%80%E0%B8%9A%E0%B8%B2%E0%B8%AB%E0%B8%A7%E0%B8%B2%E0%B8%99%E0%B8%A1%E0%B8%B5%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B8%9A%E0%B9%89%E0%B8%B2%E0%B8%87&useRAG=true"
```

## Notes

- Docker uses `/app/Modelfile.local` to create the Ollama model named `custom-model`.
- Current GGUF file: `Qwen3.5-2B.Q4_K_M.gguf`.
- Current API port: `7860`.
- `.dockerignore` excludes `.git`, local GGUF files, cache files, and docs from the build context. The GGUF model is downloaded during the Docker build by `download_models.py`.
- If Docker fails with `EOF` during export or Docker commands stop responding, restart Docker Desktop and run the build again.
