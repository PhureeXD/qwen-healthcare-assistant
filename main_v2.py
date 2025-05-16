import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="/app/chroma_db",  # C:/Users/LENOVO/Downloads/chroma_langchain_db_3
)


# Initialize ChatOllama
llm = ChatOllama(
    model="hf.co/phureexd/qwen3_v2_gguf:Q4_K_M",  # hf.co/phureexd/qwen3_v2_gguf:Q4_K_M
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    num_predict=512,
)

from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)


# retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# create cross-encoder model
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": "cpu"}
)

# Create a reranker
from langchain.retrievers.document_compressors import CrossEncoderReranker

reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# Wrap the base retriever with reranking
base_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

from langchain.retrievers import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=compression_retriever,
    name="retrieve_health_info",
    description="""
Use this tool to retrieve relevant documents from the query related to health, wellness, nutrition, exercise, symptoms, diseases, treatment, prevention, mental health, or medical advice information from the database.
Even if the query is slightly related.
Return the top 3 most relevant documents.
""",
    response_format="content_and_artifact",
)

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retriever_tool])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    # Format into prompt
    doc_infos = []
    for msg in tool_messages:
        if hasattr(msg, "artifact") and isinstance(msg.artifact, list):
            for doc in msg.artifact:  # Iterate through each Document in the artifact
                source = doc.metadata.get("source", "Unknown source")
                content = doc.page_content
                doc_infos.append(f"Source: {source}\nContent: {content}")

    docs_content = "\n\n".join(doc_infos)
    # print(f"Docs content: {docs_content}")
    system_message_content = f"""
You are a health assistant for question-answering tasks.
Use the following pieces of retrieved documents to answer the question.
If you don't know the answer, say that you don't know.
Keep the answer concise and accurate.

**Extremely important: Answer in the same language as the user query.**

### Retrieved documents (if applicable):
{docs_content}

### Few-Shot Examples:
**Example 1 (English):**
User: I feel a bit tired, what could it be?
Assistant: Fatigue can be caused by lack of sleep, stress, or dehydration. Ensure you get 7-8 hours of sleep and stay hydrated.

**Example 2 (English):**
User: Does coffee affect my health?
Assistant: Moderate coffee consumption can improve alertness but may cause insomnia or anxiety if overconsumed.

**Example 3 (English):**
User: What is the capital of France?
Assistant: The capital of France is Paris.

**Example 4 (Thai):**
User: ฉันรู้สึกเหนื่อยเล็กน้อย เกิดจากอะไรได้บ้าง?
Assistant: อาการเหนื่อยอาจเกิดจากการนอนหลับไม่เพียงพอ ความเครียด หรือภาวะขาดน้ำ ควรนอนหลับ 7-8 ชั่วโมงและดื่มน้ำให้เพียงพอ

**Example 5 (Thai):**
User: เมืองหลวงของฝรั่งเศสคืออะไร?
Assistant: เมืองหลวงของฝรั่งเศสคือปารีส

/no_think
"""
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()
# save the graph
# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


conversation_history = [
    SystemMessage(
        content="""
You are a health assistant designed to answer questions related to health, wellness, nutrition, exercise, symptoms, diseases, prevention, treatment, mental health, and medical advice. For ANY query that falls into these categories, you MUST use the retrieve_health_info tool to fetch relevant information from the database before providing an answer. This ensures your responses are accurate and based on trusted sources. Do not answer health-related questions directly without using the tool, even if you think you know the answer.

If the query is clearly unrelated to health (e.g., general knowledge questions), you can answer directly without the tool.

**Important Guidelines:**
- If the query mentions or implies health, treatment, symptoms, diseases, nutrition, exercise, mental health, or wellness, use the tool.
- Even if the query is only slightly related to health, use the tool to provide an informed answer.
- Always respond in the same language as the user's query.
- When in doubt, err on the side of using the tool.

**Examples:**

1. **Health-Related (Use Tool):**
   - User: "What are the symptoms of diabetes?"
   - Assistant: [Uses retrieve_health_info tool] "Common symptoms of diabetes include frequent urination, excessive thirst, and fatigue."

2. **Slightly Health-Related (Use Tool):**
   - User: "Is it okay to exercise when I have a cold?"
   - Assistant: [Uses retrieve_health_info tool] "Light exercise might be okay, but rest if you have a fever."

3. **Non-Health-Related (No Tool):**
   - User: "What is the capital of France?"
   - Assistant: "The capital of France is Paris."

4. **Health-Related in Thai (Use Tool):**
   - User: "อาการของโรคเบาหวานมีอะไรบ้าง?"
   - Assistant: [Uses retrieve_health_info tool] "อาการทั่วไปของโรคเบาหวาน ได้แก่ ปัสสาวะบ่อย กระหายน้ำมาก และอ่อนเพลีย"

5. **Non-Health-Related in Thai (No Tool):**
   - User: "เมืองหลวงของฝรั่งเศสคืออะไร?"
   - Assistant: "เมืองหลวงของฝรั่งเศสคือปารีส"

/no_think
"""
    )
]


# Endpoint to generate response using GET
@app.get("/generate")
async def generate(
    query: str,
    useRAG: bool = False,
    lastUserMessage: str = "",
    lastAssistantMessage: str = "",
):
    print(f"Received query: {query}")
    print(f"Last user message: {lastUserMessage}")
    print(f"Last assistant message: {lastAssistantMessage}")
    print(f"Use RAG: {useRAG}")

    if lastUserMessage:
        conversation_history.append(HumanMessage(content=lastUserMessage))
    if lastAssistantMessage:
        conversation_history.append(AIMessage(content=lastAssistantMessage))
    if useRAG:
        conversation_history.append(
            SystemMessage(
                content="""You MUST use the retrieve_health_info tool for this query."""
            )
        )

    # Append the new user query
    conversation_history.append(HumanMessage(content=query))

    def stream_response():

        for message, metadata in graph.stream(
            {
                "messages": conversation_history,
            },
            stream_mode="messages",
        ):
            # Check if the message is an AIMessage (from the generate node)
            if isinstance(message, AIMessage) and message.content:
                # Yield the content of the AIMessage
                yield f"data: {message.content}\n\n"
            # Print tool call for debugging
            elif isinstance(message, AIMessage) and message.tool_calls:
                print(f"Tool call: {message.tool_calls}")
            elif message.name == "retrieve_health_info":
                print("Retrieve:")
                print(message.artifact)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# Endpoint to clear conversation history
@app.get("/clear")
async def clear_conversation():
    global conversation_history
    conversation_history = [conversation_history[0]]  # Keep the system message

    return {"status": "success", "message": "Conversation history cleared."}


# Run the server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
