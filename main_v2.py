import os

# Suppress TensorFlow oneDNN optimization messages if not needed
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable ChromaDB telemetry to prevent log errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Set the device for HuggingFace models
device = "cpu"  # cpu

# --- Configuration Constants ---
APP_HOST = "0.0.0.0"
APP_PORT = 7860

THREAD_ID = "global_health_chat_session"  # Unique ID for the chat session

# Models and Paths
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME = "custom-model"
VECTOR_DB_PATH = "/app/chroma_db" if os.path.exists("/app/chroma_db") else "chroma_db"

# LLM Parameters
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.8
LLM_TOP_K = 20
LLM_NUM_PREDICT = 384
LLM_REPEAT_LAST_N = 256
LLM_REPEAT_PENALTY = 1.15

# Retriever Parameters
RETRIEVER_SEARCH_K = 6  # Number of documents to fetch initially
RERANKER_TOP_N = 3  # Number of documents after reranking
FORCE_RETRIEVE_MARKER = "__force_retrieve_health_info__"

HUMAN_HEALTH_KEYWORDS = (
    "ache",
    "allergy",
    "anxiety",
    "blood",
    "breath",
    "cancer",
    "chest",
    "cough",
    "covid",
    "diabetes",
    "diarrhea",
    "diet",
    "disease",
    "dizzy",
    "exercise",
    "fever",
    "flu",
    "headache",
    "health",
    "heart",
    "hospital",
    "medicine",
    "mental",
    "nausea",
    "nutrition",
    "pain",
    "pregnant",
    "sick",
    "sleep",
    "stress",
    "stroke",
    "symptom",
    "treatment",
    "vaccine",
    "vomit",
    "wellness",
    "โควิด",
    "ไข้",
    "ไอ",
    "เจ็บ",
    "เจ็บหน้าอก",
    "เชื้อ",
    "เบาหวาน",
    "แพ้",
    "แพทย์",
    "ยา",
    "รักษา",
    "วัคซีน",
    "เวียนหัว",
    "สุขภาพ",
    "หายใจ",
    "หัวใจ",
    "อาการ",
    "อาเจียน",
    "เจ็บป่วย",
    "ปวด",
    "ป่วย",
    "โรค",
    "โรงพยาบาล",
    "กระเพาะ",
    "ความดัน",
    "คลื่นไส้",
    "ตับ",
    "ติดเชื้อ",
    "ท้องผูก",
    "ท้องเสีย",
    "ปอด",
    "มะเร็ง",
    "สมอง",
    "เหนื่อย",
    "อ้วก",
    "ไต",
)

ANIMAL_HEALTH_KEYWORDS = (
    "animal",
    "cat",
    "dog",
    "pet",
    "veterinary",
    "vet",
    "หมา",
    "แมว",
    "สัตวแพทย์",
    "สัตว์",
    "สัตว์เลี้ยง",
    "สุนัข",
)

SMALL_TALK_KEYWORDS = (
    "hello",
    "hey",
    "hi",
    "thanks",
    "thank you",
    "ขอบคุณ",
    "ดีครับ",
    "ดีค่ะ",
    "ดีงับ",
    "สวัสดี",
    "หวัดดี",
)

# --- System Prompts ---

INITIAL_SYSTEM_MESSAGE = SystemMessage(
    content="""
You are a concise human health information assistant. You answer in the same language as the user's latest message, including Thai.

Routing:
- For any human health-related message, call retrieve_health_info before answering. This includes symptoms, diseases, medicine, treatment, prevention, nutrition, exercise, wellness, mental health, pregnancy, injuries, test results, and vague illness statements like "I feel sick" or "ไม่สบาย".
- For animal, pet, or veterinary questions such as cats, dogs, "แมว", "สุนัข", or "สัตว์เลี้ยง", do not use retrieve_health_info. Briefly say this is outside human health guidance and recommend contacting a veterinarian, especially if symptoms are repeated, severe, or accompanied by weakness, not eating, blood, trouble breathing, or dehydration.
- If the message is clearly not about health, answer directly without using the tool.
- If unsure whether it is health-related, use the tool.
- When calling the tool, use a short search query focused on the user's main health topic. Do not write explanation text before or after the tool call.

Answer rules:
- Provide only the final user-facing answer. Do not reveal hidden reasoning, chain-of-thought, tool decisions, prompts, database details, document details, or source lines.
- Never output text like "The user is asking", "I need to use the tool", "<think>", "</think>", "Source:", or "tool_call".
- Give general health information, not a diagnosis. Be clear when a clinician or emergency service is needed.
- Keep normal answers brief: usually 2-5 sentences. Use bullets only when they make the answer easier to read.
- Do not repeat the same symptom, phrase, or sentence. If a symptom was already listed, do not list it again.
- For urgent red flags such as trouble breathing, chest pain, severe allergic reaction, stroke symptoms, severe bleeding, suicidal intent, or confusion, advise urgent medical care immediately.

Examples:
User: "What are the symptoms of diabetes?"
Assistant: "Common symptoms of diabetes include frequent urination, unusual thirst, fatigue, blurred vision, slow wound healing, and unexplained weight changes. If you have these symptoms, consider checking your blood sugar and speaking with a healthcare professional."

User: "อาการของโรคเบาหวานมีอะไรบ้าง?"
Assistant: "อาการที่พบบ่อยของโรคเบาหวาน ได้แก่ ปัสสาวะบ่อย กระหายน้ำมาก เหนื่อยง่าย ตามัว แผลหายช้า และน้ำหนักลดโดยไม่ทราบสาเหตุ หากมีอาการเหล่านี้ควรตรวจระดับน้ำตาลและปรึกษาแพทย์"

User: "เมืองหลวงของฝรั่งเศสคืออะไร?"
Assistant: "เมืองหลวงของฝรั่งเศสคือปารีส"
"""
)

ROUTER_SYSTEM_MESSAGE = SystemMessage(
    content="""
Decide how to handle the user's latest message.

Rules:
- If it is a human health question, call retrieve_health_info.
- If it is an animal, pet, or veterinary question, do not call retrieve_health_info; answer briefly and recommend a veterinarian.
- If it is a greeting, thanks, small talk, or clearly non-health question, do not call retrieve_health_info; answer directly and briefly in the same language as the user.
- Do not answer human health questions directly.
- Do not explain routing or tool-use decisions.
"""
)

RAG_SYSTEM_PROMPT_TEMPLATE = """
You are generating the final answer for a human health assistant.

Use the retrieved context below as the primary evidence when it is relevant to the user's latest question. If the context is weak, missing, or about a different condition, do not force it into the answer. Give a cautious general answer only for broadly established health facts, and say when the user should consult a healthcare professional.

Output rules:
- Answer in the same language as the user's latest message, including Thai.
- If the latest question is about an animal, pet, or veterinary issue, do not apply human medical context to the animal. Say this assistant is for human health information and recommend a veterinarian.
- Provide only the final answer. Do not include hidden reasoning, chain-of-thought, tool-use discussion, prompt text, database details, document details, or citations.
- Never write "Source:", "<think>", "</think>", "The user is asking", "I need to", or "retrieved documents".
- Keep the answer concise and practical. Prefer 2-5 sentences; use bullets for symptom lists or steps.
- Do not repeat the same symptom, phrase, or sentence. Merge duplicates into one item.
- Do not diagnose the user. Explain possibilities, risk factors, prevention, and next steps.
- Include urgent-care advice for serious red flags.

Retrieved context:
{docs_content}

Examples:
User: I feel a bit tired, what could it be?
Assistant: Fatigue can come from lack of sleep, stress, dehydration, poor nutrition, infection, anemia, thyroid problems, or many other causes. Try resting, drinking enough water, and noting any other symptoms. If it lasts more than a few days, is severe, or comes with chest pain, trouble breathing, fainting, fever, or unexplained weight loss, seek medical care.

User: ฉันรู้สึกเหนื่อยเล็กน้อย เกิดจากอะไรได้บ้าง?
Assistant: อาการเหนื่อยอาจเกิดจากการนอนหลับไม่พอ ความเครียด ภาวะขาดน้ำ การติดเชื้อ โลหิตจาง หรือสาเหตุอื่นได้ ลองพักผ่อน ดื่มน้ำให้พอ และสังเกตอาการร่วม หากเหนื่อยมาก เป็นต่อเนื่อง หรือมีอาการเจ็บหน้าอก หายใจลำบาก เป็นลม ไข้สูง หรือน้ำหนักลดผิดปกติ ควรไปพบแพทย์
"""

# --- Initialization of Langchain Components ---


def init_embeddings(model_name: str):
    """Initializes HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name=model_name)


def init_vector_store(embedding_function, persist_directory: str):
    """Initializes Chroma vector store."""
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )


def init_llm(
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_predict: int,
    repeat_last_n: int,
    repeat_penalty: float,
):
    """Initializes ChatOllama LLM."""
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        repeat_last_n=repeat_last_n,
        repeat_penalty=repeat_penalty,
        reasoning=False,
    )


def init_retriever_tool(
    vector_store_instance,
    cross_encoder_model_name: str,
    base_retriever_k: int,
    reranker_top_n: int,
):
    """Initializes the retriever tool with reranking."""
    base_retriever = vector_store_instance.as_retriever(
        search_kwargs={"k": base_retriever_k}
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=cross_encoder_model_name,
        model_kwargs={"device": device},  # Specify device if needed, e.g., "cuda"
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=reranker_top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    return create_retriever_tool(
        retriever=compression_retriever,
        name="retrieve_health_info",
        description=(
            "Use this tool only for human health questions related to wellness, "
            "nutrition, exercise, symptoms, diseases, treatment, prevention, mental "
            "health, or medical advice information from the database. Do not use this "
            "tool for animal, pet, or veterinary questions. "
            f"Return the top {reranker_top_n} most relevant documents."
        ),
        response_format="content_and_artifact",  # Ensures artifact contains Document objects
    )


# Initialize components
print("Initializing Embeddings...")
embeddings = init_embeddings(EMBEDDING_MODEL_NAME)
print("Embeddings Initialized.")

print("Initializing Vector Store...")
vector_store = init_vector_store(embeddings, VECTOR_DB_PATH)
print("Vector Store Initialized.")

print("Initializing LLM...")
llm = init_llm(
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_NUM_PREDICT,
    LLM_REPEAT_LAST_N,
    LLM_REPEAT_PENALTY,
)
print("LLM Initialized.")

print("Initializing Retriever Tool...")
retriever_tool = init_retriever_tool(
    vector_store, CROSS_ENCODER_MODEL_NAME, RETRIEVER_SEARCH_K, RERANKER_TOP_N
)
print("Retriever Tool Initialized.")

# --- LangGraph Node Definitions ---


async def query_or_respond_node_logic(state: MessagesState):
    """
    Node function: Decides whether to call a tool for retrieval or respond directly.
    Binds the retriever_tool to the LLM for this decision.
    """
    query = _latest_human_content(state["messages"])
    direct_response = _direct_small_talk_response(query)
    if direct_response and not _is_human_health_query(query):
        return {"messages": [AIMessage(content=direct_response)]}

    if _should_retrieve_health_info(state["messages"]):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_health_info",
                            "args": {"query": query},
                            "id": "auto_retrieve_health_info",
                        }
                    ],
                )
            ]
        }

    response = await llm.bind_tools([retriever_tool]).ainvoke(
        [ROUTER_SYSTEM_MESSAGE, HumanMessage(content=query)]
    )
    _repair_retriever_tool_calls(response, state["messages"])
    if getattr(response, "tool_calls", None):
        response = response.model_copy(update={"content": ""})
    return {"messages": [response]}


async def generate_rag_response_node_logic(state: MessagesState):
    """
    Node function: Generates a response using retrieved documents (if any).
    """
    # Extract the most recent contiguous block of tool messages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":  # or isinstance(message, ToolMessage)
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format retrieved document content for the prompt
    doc_strings = []
    for tool_msg in tool_messages:
        # Ensure artifact is a list of Langchain Document objects
        if hasattr(tool_msg, "artifact") and isinstance(tool_msg.artifact, list):
            for doc in tool_msg.artifact:
                if hasattr(doc, "page_content") and hasattr(
                    doc, "metadata"
                ):  # Document structure check
                    source = doc.metadata.get("source", "Unknown source")
                    content = doc.page_content
                    doc_strings.append(f"Source: {source}\nContent: {content}")

    docs_content = (
        "\n\n".join(doc_strings)
        if doc_strings
        else "No relevant documents were found to answer the current question."
    )

    # Prepare messages for the generation LLM call (history + new system prompt with docs)
    # Include human messages, initial system messages, and AI responses (not tool calls)
    conversation_history_for_llm = [
        msg
        for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]

    # Construct the system prompt with retrieved documents
    current_system_prompt_content = RAG_SYSTEM_PROMPT_TEMPLATE.format(
        docs_content=docs_content
    )

    prompt_for_generation = [
        SystemMessage(content=current_system_prompt_content)
    ] + conversation_history_for_llm

    response = await llm.ainvoke(prompt_for_generation)
    return {"messages": [response]}


# --- LangGraph Graph Construction ---


def create_lang_graph(checkpointer_instance):
    """Creates and compiles the LangGraph."""
    graph_builder = StateGraph(MessagesState)

    # Define nodes
    graph_builder.add_node("query_or_respond", query_or_respond_node_logic)
    tools_node = ToolNode([retriever_tool])  # Define tool execution node
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("generate_rag_response", generate_rag_response_node_logic)

    # Define edges
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,  # Prebuilt condition to check for tool calls
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate_rag_response")
    graph_builder.add_edge("generate_rag_response", END)

    return graph_builder.compile(checkpointer=checkpointer_instance)


def _stringify_stream_content(content):
    """Return text from LangChain string or content-block streaming chunks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                text_parts.append(str(block))
        return "".join(text_parts)
    return str(content) if content else ""


def _strip_think_blocks(text):
    while True:
        start = text.find("<think>")
        if start == -1:
            return text.strip()

        end = text.find("</think>", start)
        if end == -1:
            return text[:start].strip()

        text = text[:start] + text[end + len("</think>") :]


def _clean_answer_content(content):
    return _strip_think_blocks(_stringify_stream_content(content))


def _format_sse_data(data):
    """Formats text as an SSE data event, preserving intentional newlines."""
    return "".join(f"data: {line}\n" for line in str(data).split("\n")) + "\n"


def _latest_human_content(messages):
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _stringify_stream_content(getattr(message, "content", ""))
    return ""


def _has_force_retrieve_marker(messages):
    return any(
        getattr(message, "type", None) == "system"
        and FORCE_RETRIEVE_MARKER in _stringify_stream_content(
            getattr(message, "content", "")
        )
        for message in messages
    )


def _is_animal_health_query(query):
    normalized_query = query.lower()
    return any(keyword in normalized_query for keyword in ANIMAL_HEALTH_KEYWORDS)


def _is_human_health_query(query):
    normalized_query = query.lower()
    return any(keyword in normalized_query for keyword in HUMAN_HEALTH_KEYWORDS)


def _contains_thai(text):
    return any("\u0e00" <= character <= "\u0e7f" for character in text)


def _direct_small_talk_response(query):
    normalized_query = query.strip().lower()
    if not normalized_query:
        return None

    if not any(keyword in normalized_query for keyword in SMALL_TALK_KEYWORDS):
        return None

    if _contains_thai(query):
        if "ขอบคุณ" in normalized_query:
            return "ยินดีครับ ถ้ามีคำถามเรื่องสุขภาพถามได้เลยครับ"
        return "สวัสดีครับ ถ้ามีคำถามเรื่องสุขภาพถามได้เลยครับ"

    if "thank" in normalized_query:
        return "You're welcome. If you have a health question, feel free to ask."
    return "Hello. If you have a health question, feel free to ask."


def _should_retrieve_health_info(messages):
    query = _latest_human_content(messages)
    if not query or _is_animal_health_query(query):
        return False

    return _has_force_retrieve_marker(messages) or _is_human_health_query(query)


def _repair_retriever_tool_calls(response, messages):
    fallback_query = _latest_human_content(messages)
    if not fallback_query:
        return

    for tool_call in getattr(response, "tool_calls", []) or []:
        if not isinstance(tool_call, dict):
            continue
        if tool_call.get("name") != "retrieve_health_info":
            continue

        args = tool_call.get("args")
        if not isinstance(args, dict):
            tool_call["args"] = {"query": fallback_query}
            continue

        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            args["query"] = fallback_query


def _iter_tool_messages(value):
    if isinstance(value, ToolMessage):
        yield value
    elif isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_tool_messages(nested_value)
    elif isinstance(value, (list, tuple, set)):
        for nested_value in value:
            yield from _iter_tool_messages(nested_value)


def _iter_ai_messages(value):
    if isinstance(value, AIMessage):
        yield value
    elif isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_ai_messages(nested_value)
    elif isinstance(value, (list, tuple, set)):
        for nested_value in value:
            yield from _iter_ai_messages(nested_value)


def _source_list_from_tool_message(tool_message: ToolMessage):
    source_list = set()
    if tool_message.name == "retrieve_health_info" and hasattr(
        tool_message, "artifact"
    ):
        print(f"Tool '{tool_message.name}' executed. Artifact content:")
        if tool_message.artifact and isinstance(tool_message.artifact, list):
            for doc in tool_message.artifact:
                if not hasattr(doc, "metadata") or not hasattr(doc, "page_content"):
                    continue

                source = doc.metadata.get("source", "Unknown source")

                if source != "Unknown source":
                    source_list.add(source)

                print(f"  Source: {source}\n   Content: {doc.page_content}")

    return source_list


# Initialize checkpointer and compile graph
memory_saver = MemorySaver()
graph = create_lang_graph(memory_saver)

# Optional: Save graph visualization
# try:
#     graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
#     print("Graph visualization saved to graph.png")
# except Exception as e:
#     print(f"Could not save graph visualization: {e}")


# --- FastAPI Application Setup ---
app = FastAPI(
    title="Health Assistant API",
    description="API for a health assistant using a retrieval-augmented generation approach.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FastAPI Endpoints ---


@app.get("/", summary="Root endpoint")
async def root():
    return {"message": "Health Assistant API is running!", "docs_url": "/docs"}


@app.get("/generate", summary="Generate a response from the health assistant model")
async def generate_endpoint(
    query: str,
    useRAG: bool = False,
    thread_id: str = THREAD_ID,
):
    """
    Handles a user query, streams back the assistant's responses.
    - `query`: The user's question.
    - `useRAG`: If true, forces the use of the retrieval tool via a system message.
    - `thread_id`: Unique identifier for the conversation session.
    """
    print(f"Received query: '{query}', Force RAG: {useRAG}, Thread ID: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    # Prepare input messages for the graph
    input_messages = []
    current_checkpoint_tuple = memory_saver.get_tuple(config)  # Check if history exists

    if current_checkpoint_tuple is None:  # No history, it's a new or cleared session
        input_messages.append(INITIAL_SYSTEM_MESSAGE)
        print("Starting new conversation: Added initial system message.")

    if useRAG:
        # This message is added to strongly encourage tool use for the current query,
        # supplementing the INITIAL_SYSTEM_MESSAGE.
        input_messages.append(
            SystemMessage(
                content=FORCE_RETRIEVE_MARKER
            )
        )
        print("Forcing RAG for this query with an additional system message.")

    input_messages.append(HumanMessage(content=query))
    graph_input = {"messages": input_messages}

    async def stream_response_events():
        async for chunk in graph.astream(graph_input, config, stream_mode="updates"):
            if not chunk:
                continue

            for tool_message in _iter_tool_messages(chunk):
                source_list = _source_list_from_tool_message(tool_message)
                yield _format_sse_data(f"**Source:**{str(source_list)}\n")

            for ai_message in _iter_ai_messages(chunk):
                if getattr(ai_message, "tool_calls", None):
                    print(f"AI requested Tool call: {ai_message.tool_calls}")
                    continue

                content = _clean_answer_content(getattr(ai_message, "content", ""))
                if content:
                    yield _format_sse_data(content)

    return StreamingResponse(
        stream_response_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/clear", summary="Clear conversation history")
async def clear_conversation_endpoint(thread_id: str = THREAD_ID):
    """Clears the conversation history for the specified thread_id."""
    try:
        memory_saver.delete_thread(thread_id)
        print(f"Conversation history cleared for thread_id: {thread_id}")
        return {"status": "success", "message": "Conversation history cleared."}
    except Exception as e:
        print(f"Error clearing conversation history for thread_id {thread_id}: {e}")
        return {"status": "error", "message": f"Failed to clear history: {e}"}


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Health Assistant API on {APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
