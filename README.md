---
title: Phuree Healthcare Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Qwen Healthcare Assistant

This is a healthcare assistant powered by a custom Qwen model.

## Current Runtime Flow

The Docker app runs `main_v2:app` with FastAPI and exposes:

- `GET /` for a health check.
- `GET /generate?query=...&useRAG=false&thread_id=...` for SSE chat streaming.
- `GET /clear?thread_id=...` to clear one conversation thread.

### Request Path

```text
Client
  -> GET /generate?query=...&useRAG=...&thread_id=...
  -> FastAPI generate_endpoint()
  -> Load LangGraph memory by thread_id
  -> If new thread: add initial health-assistant system message
  -> If useRAG=true: add internal force-retrieval marker
  -> Add user query
  -> Run LangGraph
  -> Stream SSE response back to client
```

### Routing Path

```text
User query
  -> Pure small talk?
       -> Yes: direct short reply
       -> No:
  -> Health capability/meta question?
       -> Yes: direct scope reply
       -> No:
  -> Clear animal or pet-health question?
       -> Yes: direct outside-scope reply
       -> No:
  -> Clear human-health question?
       -> Yes: retrieve_health_info tool call
       -> No:
  -> LLM router classifies query
       -> RETRIEVE: retrieve_health_info tool call
       -> ANIMAL: direct outside-scope reply
       -> DIRECT: direct response with conversation history
```

### RAG Path

```text
retrieve_health_info
  -> Chroma vector search
  -> BAAI/bge-m3 embeddings
  -> Candidate documents
  -> BAAI/bge-reranker-v2-m3 cross-encoder rerank
  -> Top reranked documents
  -> RAG prompt with relevant context
  -> Qwen/Ollama answer
  -> Clean <think>...</think>
  -> Stream:
       **Source:**{...}

       final answer
```

### Direct Path

```text
DIRECT route
  -> Build prompt from:
       DIRECT_RESPONSE_SYSTEM_MESSAGE
       + prior human messages
       + prior AI messages without tool calls
  -> Qwen/Ollama answer
  -> Clean <think>...</think>
  -> Stream final answer
```

### Error Path

```text
Ollama / router / RAG / streaming failure
  -> Catch exception
  -> Log error on server
  -> Stream fallback message to client
  -> Avoid raw 500 traceback in chat UI
```
