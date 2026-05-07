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
device = "cpu"

# --- Configuration Constants ---
APP_HOST = "0.0.0.0"
APP_PORT = 7860

THREAD_ID = "global_health_chat_session"  # Unique ID for the chat session

# Models and Paths
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME = "custom-model"  # Replace with your actual model, e.g., "hf.co/phureexd/qwen3_v2_gguf:Q4_K_M"
VECTOR_DB_PATH = "/app/chroma_db" if os.path.exists("/app/chroma_db") else "chroma_db"

# LLM Parameters
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
LLM_TOP_K = 20
LLM_NUM_PREDICT = 512

# Retriever Parameters
RETRIEVER_SEARCH_K = 6  # Number of documents to fetch initially
RERANKER_TOP_N = 3  # Number of documents after reranking

# --- System Prompts ---

INITIAL_SYSTEM_MESSAGE = SystemMessage(
    content="""
You are a health assistant designed to answer questions related to health, wellness, nutrition, exercise, symptoms, diseases, prevention, treatment, mental health, and medical advice. This explicitly includes general statements about feeling unwell or sick (e.g., "I'm sick", "I don't feel good"). For ANY query that falls into these categories, you MUST use the retrieve_health_info tool to fetch relevant information from the database before providing an answer. This ensures your responses are accurate and based on trusted sources. Do not answer health-related questions directly without using the tool, even if you think you know the answer.

If the query is clearly unrelated to health (e.g., general knowledge questions), you can answer directly without the tool.

**Important Guidelines:**
- If the query mentions or implies health, feeling unwell, sickness, treatment, symptoms, diseases, nutrition, exercise, mental health, or wellness, use the tool.
- Even if the query is only slightly related to health, or is a general statement about feeling unwell, use the tool to provide an informed answer.
- Always respond in the same language as the user's query.
- When in doubt, err on the side of using the tool.

**Examples:**

1. **Health-Related (Use Tool):**
   - User: "What are the symptoms of diabetes?"
   - Assistant: [Uses retrieve_health_info tool] "Common symptoms of diabetes include frequent urination, excessive thirst, and fatigue."

2. **Slightly Health-Related (Use Tool):**
   - User: "Is it okay to exercise when I have a cold?"
   - Assistant: [Uses retrieve_health_info tool] "Light exercise might be okay, but rest if you have a fever."

3. **General Sickness Statement (Use Tool):**
   - User: "I'm sick."
   - Assistant: [Uses retrieve_health_info tool] "I'm sorry to hear you're not feeling well. Common advice includes resting and staying hydrated. If you have specific symptoms, I can try to provide more information."

4. **Non-Health-Related (No Tool):**
   - User: "What is the capital of France?"
   - Assistant: "The capital of France is Paris."

5. **Health-Related in Thai (Use Tool):**
   - User: "อาการของโรคเบาหวานมีอะไรบ้าง?"
   - Assistant: [Uses retrieve_health_info tool] "อาการทั่วไปของโรคเบาหวาน ได้แก่ ปัสสาวะบ่อย กระหายน้ำมาก และอ่อนเพลีย"

6. **Non-Health-Related in Thai (No Tool):**
   - User: "เมืองหลวงของฝรั่งเศสคืออะไร?"
   - Assistant: "เมืองหลวงของฝรั่งเศสคือปารีส"
/no_think                                       
"""
)

RAG_SYSTEM_PROMPT_TEMPLATE = """
You are a health assistant for question-answering tasks.
Use the following pieces of retrieved documents to answer the question.
If you don't know the answer, say that you don't know.
Keep the answer concise and accurate.

**Extremely important: Answer in the same language as the user query.**

### Retrieved documents (if applicable):
{docs_content}

### Examples of the language model's responses:
**Example 1 (English):**
User: I feel a bit tired, what could it be?
Assistant: Fatigue can be caused by lack of sleep, stress, or dehydration. Ensure you get 7-8 hours of sleep and stay hydrated.

**Example 2 (English):**
User: Does coffee affect my health?
Assistant: Moderate coffee consumption can improve alertness but may cause insomnia or anxiety if overconsumed.

**Example 3 (Thai):**
User: ฉันรู้สึกเหนื่อยเล็กน้อย เกิดจากอะไรได้บ้าง?
Assistant: อาการเหนื่อยอาจเกิดจากการนอนหลับไม่เพียงพอ ความเครียด หรือภาวะขาดน้ำ ควรนอนหลับ 7-8 ชั่วโมงและดื่มน้ำให้เพียงพอ
/no_think
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
    model_name: str, temperature: float, top_p: float, top_k: int, num_predict: int
):
    """Initializes ChatOllama LLM."""
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
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
            "Use this tool to retrieve relevant documents from the query related to health, "
            "wellness, nutrition, exercise, symptoms, diseases, treatment, prevention, "
            "mental health, or medical advice information from the database. "
            "Even if the query is slightly related. "
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
llm = init_llm(LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_NUM_PREDICT)
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
    response = await llm.bind_tools([retriever_tool]).ainvoke(state["messages"])
    _repair_retriever_tool_calls(response, state["messages"])
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


def _latest_human_content(messages):
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _stringify_stream_content(getattr(message, "content", ""))
    return ""


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
                content="You MUST use the retrieve_health_info tool for this query even if it seems unrelated to health."
            )
        )
        print("Forcing RAG for this query with an additional system message.")

    input_messages.append(HumanMessage(content=query))
    graph_input = {"messages": input_messages}

    async def stream_response_events():
        async for stream_mode, chunk in graph.astream(
            graph_input, config, stream_mode=["messages", "updates"]
        ):
            if not chunk:
                continue

            if stream_mode == "messages":
                message_chunk, _metadata = chunk
                if isinstance(message_chunk, AIMessage):
                    content = _stringify_stream_content(
                        getattr(message_chunk, "content", "")
                    )
                    if content:
                        yield f"data: {content}\n\n"
                    elif getattr(message_chunk, "tool_calls", None):
                        print(f"AI requested Tool call: {message_chunk.tool_calls}")
                continue

            if stream_mode == "updates":
                for tool_message in _iter_tool_messages(chunk):
                    source_list = _source_list_from_tool_message(tool_message)
                    yield f"data: **Source:**{str(source_list)}\n\n"

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
