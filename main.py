import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from unsloth import FastLanguageModel

print()
import time
from threading import Thread

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
###########################################################################
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

from langchain_chroma import Chroma

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="C:/Users/LENOVO/Downloads/chroma_langchain_db_3",  # Where to save data locally, remove if not necessary
)


# Model configuration
model_name = "phureexd/qwen_model"
max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)


# Step 3: Prepare RAG messages
def prepare_rag_messages(messages, vector_store, k=2):
    query = next(msg["content"] for msg in reversed(messages) if msg["role"] == "user")
    print("this is query:\n", type(query), query)
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join(
        f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs
    )
    print("this is context:\n", context)
    system_message = messages[0]["content"] + "\n\nContext:\n" + context
    rag_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]
    return rag_messages


# Endpoint to generate response using GET
@app.get("/generate")
async def generate(query: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health assistant. Use the provided context to answer the user question. "
                "If the context is unrelated, answer without it. Keep the answer concise."
            ),
        },
        {"role": "user", "content": f"{query}"},
    ]

    rag_messages = prepare_rag_messages(messages, vector_store, k=2)

    text = tokenizer.apply_chat_template(
        rag_messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(device=model.device)

    def stream_response():
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        # * the recommended settings for reasoning inference are temperature = 0.6, top_p = 0.95, top_k = 20
        # * For normal chat based inference, temperature = 0.7, top_p = 0.8, top_k = 20
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.daemon = True
        thread.start()

        for new_text in streamer:
            yield f"data: {new_text}\n\n"
            # time.sleep(0.01)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
