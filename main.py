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
# model_name = "phureexd/qwen_model"
model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
max_seq_length = 2048
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
            "content": f"""You are a medical professional assistant. You will receive user queries along with relevant context retrieved via RAG.
Use the context if it is relevant. If not, rely on your own medical knowledge. If unsure, clearly state so.
Always respond in the same language used in the user's query. Keep responses clear, concise, and professional.

One-Shot Example / ตัวอย่างการตอบ:

User (Thai):
ฉันมีอาการเวียนหัวตอนตื่นนอน เกิดจากอะไรได้บ้าง?

Context (if any):
เวียนหัวตอนตื่นนอนอาจเกิดจากความดันเลือดต่ำเมื่อเปลี่ยนท่าทาง หรือภาวะน้ำในหูไม่เท่ากัน

Assistant (Thai):
อาการเวียนหัวตอนตื่นนอนอาจเกิดจากภาวะความดันโลหิตต่ำเมื่อเปลี่ยนท่าทางอย่างรวดเร็ว (Orthostatic hypotension) หรืออาจเกี่ยวข้องกับระบบการทรงตัวในหูชั้นใน เช่น ภาวะน้ำในหูไม่เท่ากัน หากอาการเป็นบ่อยหรือรุนแรง ควรพบแพทย์เพื่อตรวจเพิ่มเติม

User (English):
I often feel dizzy after standing up. Is this something serious?

Context (if any):
Dizziness after standing may be due to orthostatic hypotension, a drop in blood pressure when changing position.

Assistant (English):
Feeling dizzy after standing up can be caused by orthostatic hypotension, which is a drop in blood pressure due to a sudden posture change. It's usually not dangerous but if it happens frequently or is accompanied by fainting, it's best to consult a healthcare provider.
""",
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
        # generate_kwargs = dict(
        #     **inputs,
        #     max_new_tokens=2048,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.95,
        #     top_k=20,
        #     streamer=streamer,
        # )
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        # thread.daemon = True
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
