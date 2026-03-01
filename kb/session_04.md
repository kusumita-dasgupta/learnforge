
Agentic RAG From Scratch: Building with LangGraph and Open-Source Models
In this notebook, we'll look under the hood of create_agent and build an agentic RAG application from scratch using LangGraph's low-level primitives and locally-hosted open-source models.

Learning Objectives:

Understand LangGraph's core constructs: StateGraph, nodes, edges, and conditional routing
Build a ReAct agent from scratch without high-level abstractions
Use Ollama to run open-source models locally (gpt-oss:20b + embeddinggemma)
Transition from aimakerspace utilities to the LangChain ecosystem
Table of Contents:
Breakout Room #1: LangGraph Fundamentals & Building Agents from Scratch

Task 1: Dependencies & Ollama Setup
Task 2: LangGraph Core Concepts (StateGraph, Nodes, Edges)
Task 3: Building a ReAct Agent from Scratch
Task 4: Adding Tools to Your Agent
Question #1 & Question #2
Activity #1: Implement a Custom Routing Function
Breakout Room #2: Agentic RAG with Local Models

Task 5: Loading & Chunking with LangChain
Task 6: Setting up Qdrant with Local Embeddings
Task 7: Creating a RAG Tool
Task 8: Building Agentic RAG from Scratch
Question #3 & Question #4
Activity #2: Extend the Agent with Memory
Breakout Room #1
LangGraph Fundamentals & Building Agents from Scratch
Task 1: Dependencies & Ollama Setup
Before we begin, make sure you have:

Ollama installed - Download from ollama.com
Ollama running - Start with ollama serve in a terminal
Models pulled - Run these commands:
# Chat model for reasoning and generation (~12GB)
ollama pull gpt-oss:20b

# Embedding model for RAG (~622MB)
ollama pull embeddinggemma
Note: If you don't have enough RAM/VRAM for gpt-oss:20b (requires 16GB+ VRAM or 24GB+ RAM), you can substitute with llama3.2:3b or another smaller model.

📚 Documentation:

Ollama Installation Guide
gpt-oss Model Card
EmbeddingGemma Model Card
langchain-ollama Integration
# Core imports we'll use throughout the notebook
import os
import getpass
import json
from uuid import uuid4
from typing import Annotated, TypedDict, Literal

import nest_asyncio
nest_asyncio.apply()  # Required for async operations in Jupyter
# Verify Ollama is running and models are available
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Test connection to Ollama
try:
    #test_llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    test_llm = ChatOllama(model="llama3.2:3b", temperature=0) 
    test_response = test_llm.invoke("Say 'Ollama is working!' in exactly 3 words.")
    print(f"Chat Model Test: {test_response.content}")
    
    test_embeddings = OllamaEmbeddings(model="embeddinggemma")
    test_vector = test_embeddings.embed_query("test")
    print(f"Embedding Model Test: Vector dimension = {len(test_vector)}")
    print("\nOllama is ready!")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("\nMake sure:")
    print("1. Ollama is installed: https://ollama.com/")
    print("2. Ollama is running: 'ollama serve'")
    #print("3. Models are pulled: 'ollama pull gpt-oss:20b' and 'ollama pull embeddinggemma'")
    print("3. Models are pulled: 'ollama pull llama3.2:3b' and 'ollama pull embeddinggemma'")
Chat Model Test: "Ollama is working!"
Embedding Model Test: Vector dimension = 768

Ollama is ready!
Task 2: LangGraph Core Concepts
In Session 3, we used create_agent which abstracts away the complexity. Now let's understand what's happening under the hood!

LangGraph models workflows as graphs with three key components:
1. State
A shared data structure that represents the current snapshot of your application:

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
The add_messages reducer ensures new messages are appended (not replaced) when the state updates.

2. Nodes
Python functions that encode the logic of your agent:

Receive the current state
Perform computation or side-effects
Return an updated state
3. Edges
Functions that determine which node to execute next:

Normal edges: Always go to a specific node
Conditional edges: Choose the next node based on state
📚 Documentation:

LangGraph Low-Level Concepts
LangGraph Quickstart
StateGraph API Reference
# Let's build our first LangGraph workflow - a simple echo graph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# Step 1: Define the State
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]

# Step 2: Define Nodes (functions that process state)
def echo_node(state: SimpleState):
    """A simple node that echoes the last message."""
    last_message = state["messages"][-1]
    echo_response = AIMessage(content=f"You said: {last_message.content}")
    return {"messages": [echo_response]}

# Step 3: Build the Graph
echo_graph = StateGraph(SimpleState)

# Add nodes
echo_graph.add_node("echo", echo_node)

# Add edges (START -> echo -> END)
echo_graph.add_edge(START, "echo")
echo_graph.add_edge("echo", END)

# Compile the graph
echo_app = echo_graph.compile()

print("Simple echo graph created!")
Simple echo graph created!
# Visualize the graph structure
try:
    from IPython.display import display, Image
    display(Image(echo_app.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph image: {e}")
    print("\nGraph structure (ASCII):")
    print(echo_app.get_graph().draw_ascii())

# Test the echo graph
result = echo_app.invoke({"messages": [HumanMessage(content="Hello, LangGraph!")]})

print("Conversation:")
for msg in result["messages"]:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{role}]: {msg.content}")
Conversation:
  [Human]: Hello, LangGraph!
  [AI]: You said: Hello, LangGraph!
Task 3: Building a ReAct Agent from Scratch
Now let's build something more sophisticated: a ReAct agent that can:

Reason about what to do
Act by calling tools
Observe results
Repeat until done
This is exactly what create_agent does under the hood. Let's build it ourselves!

The Agent Loop Architecture
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
             ┌─────►│    agent     │◄────────┐
             │      │  (call LLM)  │         │
             │      └──────┬───────┘         │
             │             │                 │
             │             ▼                 │
             │      ┌──────────────┐         │
             │      │ should_      │         │
             │      │ continue?    │         │
             │      └──────┬───────┘         │
             │             │                 │
             │    tool_calls?                │
             │     │           │             │
             │    YES         NO             │
             │     │           │             │
             │     ▼           ▼             │
             │ ┌────────┐  ┌───────┐         │
             │ │ tools  │  │  END  │         │
             └─┤(execute│  └───────┘         │
               │ tools) ├────────────────────┘
               └────────┘
📚 Documentation:

How to create a ReAct agent from scratch
ReAct Agent Conceptual Guide
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# Step 1: Define the Agent State
class AgentState(TypedDict):
    """The state of our agent - just a list of messages."""
    messages: Annotated[list[BaseMessage], add_messages]

print("AgentState defined with messages field")
AgentState defined with messages field
# Step 2: Initialize our local LLM with Ollama
llm = ChatOllama(
    #model="gpt-oss:20b",
    model="llama3.2:3b", 
    temperature=0,  # Deterministic for reproducibility
)

print(f"LLM initialized: {llm.model}")
LLM initialized: llama3.2:3b
Task 4: Adding Tools to Your Agent
Tools are functions that the agent can call. We use the @tool decorator and bind them to the LLM.

📚 Documentation:

LangChain Tools Conceptual Guide
@tool Decorator Reference
ToolNode Prebuilt
# Step 3: Define Tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')
    """
    try:
        # Using eval with restricted globals for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks about the current time or date."""
    from datetime import datetime
    return f"The current date and time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Create our tool list
tools = [calculate, get_current_time]

# Bind tools to the LLM - this tells the LLM about available tools
llm_with_tools = llm.bind_tools(tools)

print("Tools defined and bound to LLM:")
for t in tools:
    print(f"  - {t.name}: {t.description[:50]}...")
Tools defined and bound to LLM:
  - calculate: Evaluate a mathematical expression. Use this for a...
  - get_current_time: Get the current date and time. Use this when the u...
# Step 4: Define the Agent Node (calls the LLM)
SYSTEM_PROMPT = """You are a helpful assistant that can perform calculations and tell the time.
Always use the available tools when appropriate.
Be concise in your responses."""

def agent_node(state: AgentState):
    """The agent node - calls the LLM with the current conversation."""
    # Prepare messages with system prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # Call the LLM
    response = llm_with_tools.invoke(messages)
    
    # Return the response to be added to state
    return {"messages": [response]}

print("Agent node defined")
Agent node defined
# Step 5: Define the Tool Node (executes tools)
# We can use LangGraph's prebuilt ToolNode for convenience
tool_node = ToolNode(tools)

print("Tool node created using ToolNode prebuilt")
Tool node created using ToolNode prebuilt
# Step 6: Define the Conditional Edge (routing logic)
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to call tools or end the conversation."""
    last_message = state["messages"][-1]
    
    # If the LLM made tool calls, route to tools node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"

print("Conditional routing function defined")
Conditional routing function defined
# Step 7: Build the Graph!
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.add_edge(START, "agent")

# Add conditional edge from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # If should_continue returns "tools", go to tools node
        "end": END         # If should_continue returns "end", finish
    }
)

# Add edge from tools back to agent (the loop!)
workflow.add_edge("tools", "agent")

# Compile the graph
agent = workflow.compile()

print("ReAct agent built from scratch!")
ReAct agent built from scratch!
# Visualize our agent
try:
    from IPython.display import display, Image
    display(Image(agent.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph image: {e}")
    print("\nGraph structure (ASCII):")
    print(agent.get_graph().draw_ascii())

# Test our agent!
print("Testing our from-scratch agent:")
print("=" * 50)

response = agent.invoke({"messages": [HumanMessage(content="What is 25 * 48?")]})

print("\nConversation:")
for msg in response["messages"]:
    msg_type = type(msg).__name__
    content = msg.content if msg.content else f"[Tool calls: {msg.tool_calls}]" if hasattr(msg, 'tool_calls') and msg.tool_calls else "[No content]"
    print(f"  [{msg_type}]: {content[:200]}")
Testing our from-scratch agent:
==================================================

Conversation:
  [HumanMessage]: What is 25 * 48?
  [AIMessage]: [Tool calls: [{'name': 'calculate', 'args': {'expression': '25 * 48'}, 'id': '6ab9e332-718b-4e56-aa74-66478de2ee54', 'type': 'tool_call'}]]
  [ToolMessage]: The result of 25 * 48 is 1200
  [AIMessage]: 1200
# Test with multiple tools
print("Testing with multiple tool calls:")
print("=" * 50)

response = agent.invoke({
    "messages": [HumanMessage(content="What time is it, and what is 100 divided by the current hour?")]
})

print("\nFinal response:")
print(response["messages"][-1].content)
Testing with multiple tool calls:
==================================================

Final response:
I apologize for the earlier error. Since I'm a text-based AI assistant, I don't have real-time access to the current time. However, I can tell you that the hour is 12.

Now, let's calculate 100 divided by 12:

100 / 12 = 8.33
# Stream the agent's execution to see it step by step
print("Streaming agent execution:")
print("=" * 50)

for chunk in agent.stream(
    {"messages": [HumanMessage(content="Calculate 15% of 200")]},
    stream_mode="updates"
):
    for node_name, values in chunk.items():
        print(f"\n[Node: {node_name}]")
        if "messages" in values:
            for msg in values["messages"]:
                if hasattr(msg, 'content') and msg.content:
                    print(f"  Content: {msg.content[:200]}")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"  Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
Streaming agent execution:
==================================================

[Node: agent]
  Tool calls: ['calculate']

[Node: tools]
  Content: The result of 0.15 * 200 is 30.0

[Node: agent]
  Content: 15% of 200 is 30.
❓ Question #1:
In our from-scratch agent, we defined a should_continue function that returns either "tools" or "end". How does this compare to how create_agent handles the same decision? What additional logic might create_agent include that we didn't implement?

Answer:
In the from-scratch agent, should_continue() is an explicit, minimal router: it checks the last AI message and returns "tools" if tool_calls exist, otherwise "end". create_agent makes the same decision implicitly, but usually adds production guardrails you didn’t implement, such as: i. Iteration/loop limits to prevent infinite tool loops. ii. Tool-call validation/normalization (unknown tool names, bad args/schema, empty tool calls). iii. Tool error handling + retry/fallback instead of crashing or ending silently. iv. State/message hygiene (ensures ToolMessages are appended correctly and conversation state stays consistent). v. Optional safety/policy gates (block certain tools, require approval/HITL). vi. Observability hooks (tracing/callbacks/streaming, token/cost accounting).

❓ Question #2:
We used ToolNode from langgraph.prebuilt to execute tools. Looking at the tool execution flow, what would happen if we wanted to add logging, error handling, or rate limiting to tool execution? How would building our own tool node give us more control?

Answer:
Using ToolNode is convenient, but it’s a black box: it will run whatever tool(s) the model requested and append the ToolMessage results back into state["messages"]. If you want logging, custom error handling, or rate limiting, you either have to rely on whatever defaults ToolNode provides, or add cross-cutting hooks outside it, you can’t precisely control the per-tool execution flow.

Building your own tool node (a custom node function) gives you that control because you explicitly iterate over last_message.tool_calls and decide what to do for each call. That enables you to: i. Logging/telemetry: log tool name, args (redacted), start/end timestamps, latency, success/failure, and how often each tool is called. ii. Error handling: wrap each tool call in try/except, return a structured “tool failed” ToolMessage (instead of crashing), and optionally trigger retries or fallback behavior. iii. Rate limiting / quotas: enforce per-tool or global limits (e.g., max calls per run, per minute, per user/thread), add backoff, or block calls when limits are hit. iv. Policy checks: allow/deny tools based on user role, tool risk, or content (e.g., block “web” tools, require approval for “write” actions). v. Tool routing: send different tool types to different nodes (e.g., search_wellness_knowledge vs calculate) or run some tools in sequence/parallel. vi. Argument validation: enforce schemas, clamp inputs, sanitize strings, and reject malformed calls before executing. vii. Caching/dedup: avoid repeated identical tool calls in the same run and reuse results.

So: ToolNode = fast default execution. Custom tool node = full control over execution, safety, observability, and reliability.

🏗️ Activity #1: Implement a Custom Routing Function
Extend the agent by implementing a custom routing function that adds more sophisticated logic.

Ideas:

Add a maximum iteration limit to prevent infinite loops
Route to different nodes based on the type of tool being called
Add a "thinking" step before tool execution
Requirements:

Modify the should_continue function or create a new one
Add any new nodes if needed
Rebuild and test the agent
📚 Documentation:

Conditional Edges
How to create branches for parallel node execution
Add a “think” step before tools: Explicitly block tool execution, with graceful fallback
Step1 : Extend state with a flag

### YOUR CODE HERE ###

# Example: Add iteration tracking to prevent infinite loops
class AgentStateWithCounter(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int

def custom_should_continue(state: AgentStateWithCounter) -> Literal["tools", "end"]:
    """Custom routing with iteration limit."""
    # Your implementation here
    pass

# Build your custom agent
# Step 1: Extend state with a flag
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# NOTE: This assumes you already defined these earlier in the notebook:
# - agent_node (the LLM-calling node)
# - tool_node  (ToolNode(tools))

class AgentStateWithPlan(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    allow_tools: bool  # set by think_node
Step 2: Add a “think” node (policy/governor)

This node does not call tools. It inspects the last AI message and decides if tool calls should run.

# Step 2: Add a “think” node (policy/governor) that blocks tools
def think_node(state: AgentStateWithPlan):
    last = state["messages"][-1]

    # Explicit policy: block all tool usage
    allow_tools = False

    # (Optional) show what the model *wanted* to call
    if hasattr(last, "tool_calls") and last.tool_calls:
        print("Model requested tools:", [tc["name"] for tc in last.tool_calls])

    return {"allow_tools": allow_tools}
Step 3: Add a router after “think”

# Step 3: Router after “think”
# If tools are blocked, route to fallback (instead of ending silently)
def route_after_think(state: AgentStateWithPlan) -> Literal["tools", "fallback"]:
    return "tools" if state.get("allow_tools", False) else "fallback"
Step 4: Add a fallback node (graceful behavior when tools are blocked)

# Step 4: Add a fallback node (graceful behavior when tools are blocked)
def fallback_node(state: AgentStateWithPlan):
    last = state["messages"][-1]
    requested = []
    if hasattr(last, "tool_calls") and last.tool_calls:
        requested = [tc["name"] for tc in last.tool_calls]

    msg = AIMessage(
        content=(
            f"I can’t run tools right now (blocked by policy). "
            f"Requested tool(s): {requested}. "
            f"I can still answer without tools—do you want me to proceed?"
        )
    )
    return {"messages": [msg]}
Step 5: Rebuild graph: agent → think → (tools | fallback ) → agent/end

# Step 5: Rebuild graph: agent → think → (tools | fallback) → agent/end
workflow2 = StateGraph(AgentStateWithPlan)

workflow2.add_node("agent", agent_node)     # from ReAct section
workflow2.add_node("think", think_node)
workflow2.add_node("tools", tool_node)      # from ReAct section
workflow2.add_node("fallback", fallback_node)

workflow2.add_edge(START, "agent")
workflow2.add_edge("agent", "think")

workflow2.add_conditional_edges(
    "think",
    route_after_think,
    {"tools": "tools", "fallback": "fallback"}
)

# If tools run, loop back to agent; fallback ends
workflow2.add_edge("tools", "agent")
workflow2.add_edge("fallback", END)

agent_with_blocked_tools = workflow2.compile()
Step 6: Test

# ---- Test ----
resp = agent_with_blocked_tools.invoke(
    {"messages": [HumanMessage(content="What is 25 * 48?")], "allow_tools": True}
)

print("Final message:", resp["messages"][-1].content)
Model requested tools: ['calculate']
Final message: I can’t run tools right now (blocked by policy). Requested tool(s): ['calculate']. I can still answer without tools—do you want me to proceed?
Breakout Room #2
Agentic RAG with Local Models
Now let's build a full Agentic RAG system from scratch using our local models!

We'll transition from the aimakerspace utilities to the LangChain ecosystem:

Task	aimakerspace	LangChain
Load Documents	TextFileLoader	TextLoader
Split Text	CharacterTextSplitter	RecursiveCharacterTextSplitter
Embeddings	Custom	OllamaEmbeddings
Task 5: Loading & Chunking with LangChain
Let's use LangChain's document loaders and text splitters.

📚 Documentation:

Document Loaders Conceptual Guide
TextLoader Reference
RecursiveCharacterTextSplitter
Text Splitters Conceptual Guide
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document using LangChain's TextLoader
loader = TextLoader("data/HealthWellnessGuide.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Total characters: {sum(len(doc.page_content) for doc in documents):,}")
print(f"\nDocument metadata: {documents[0].metadata}")
Loaded 1 document(s)
Total characters: 16,206

Document metadata: {'source': 'data/HealthWellnessGuide.txt'}
# Split documents using RecursiveCharacterTextSplitter
# This is more sophisticated than simple character splitting!

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    # Default separators: ["\n\n", "\n", " ", ""]
    # Tries to keep paragraphs, then sentences, then words together
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print(f"\nSample chunk (first 300 chars):")
print("-" * 50)
print(chunks[0].page_content[:300] + "...")
Split into 45 chunks

Sample chunk (first 300 chars):
--------------------------------------------------
The Personal Wellness Guide
A Comprehensive Resource for Health and Well-being

PART 1: EXERCISE AND MOVEMENT

Chapter 1: Understanding Exercise Basics

Exercise is one of the most important things you can do for your health. Regular physical activity can improve your brain health, help manage weigh...
Task 6: Setting up Qdrant with Local Embeddings
Now we'll use OllamaEmbeddings with the embeddinggemma model - completely local!

📚 Documentation:

OllamaEmbeddings Reference
Qdrant Vector Store Integration
Embedding Models Conceptual Guide
EmbeddingGemma Overview (Google)
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Initialize local embedding model
embedding_model = OllamaEmbeddings(model="embeddinggemma")

# Get embedding dimension
sample_embedding = embedding_model.embed_query("test")
embedding_dim = len(sample_embedding)
print(f"Embedding dimension: {embedding_dim}")
print(f"Using local model: embeddinggemma")
Embedding dimension: 768
Using local model: embeddinggemma
# Create Qdrant client (in-memory for development)
qdrant_client = QdrantClient(":memory:")

# Create a collection for our wellness documents
collection_name = "wellness_knowledge_base_local"

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embedding_dim,
        distance=Distance.COSINE
    )
)

print(f"Created collection: {collection_name}")
Created collection: wellness_knowledge_base_local
# Create vector store and add documents
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model
)

# Add documents to the vector store
print("Adding documents to vector store (this may take a moment with local embeddings)...")
vector_store.add_documents(chunks)
print(f"Added {len(chunks)} documents to vector store")
Adding documents to vector store (this may take a moment with local embeddings)...
Added 45 documents to vector store
# Test the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

test_results = retriever.invoke("How can I improve my sleep?")

print("Retrieved documents:")
for i, doc in enumerate(test_results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:200] + "...")
Retrieved documents:

--- Document 1 ---
Chapter 8: Improving Sleep Quality

Sleep hygiene refers to habits and practices that promote consistent, quality sleep.

Essential sleep hygiene practices:
- Maintain a consistent sleep schedule, eve...

--- Document 2 ---
Creating an optimal sleep environment:
- Temperature: 65-68 degrees Fahrenheit (18-20 Celsius)
- Darkness: Use blackout curtains or a sleep mask
- Quiet: Consider white noise machines or earplugs
- Co...

--- Document 3 ---
Types of insomnia:
- Acute insomnia: Short-term, often triggered by stress or life events
- Chronic insomnia: Long-term, occurring at least 3 nights per week for 3 months or more

Natural remedies for...
Task 7: Creating a RAG Tool
Now let's wrap our retriever as a tool that the agent can use.

@tool
def search_wellness_knowledge(query: str) -> str:
    """Search the wellness knowledge base for information about health, fitness, nutrition, sleep, and mental wellness.
    
    Use this tool when the user asks questions about:
    - Physical health and fitness
    - Nutrition and diet
    - Sleep and rest
    - Mental health and stress management
    - General wellness tips
    
    Args:
        query: The search query to find relevant wellness information
    """
    results = retriever.invoke(query)
    
    if not results:
        return "No relevant information found in the wellness knowledge base."
    
    # Format the results
    formatted_results = []
    for i, doc in enumerate(results, 1):
        formatted_results.append(f"[Source {i}]:\n{doc.page_content}")
    
    return "\n\n".join(formatted_results)

print(f"RAG tool created: {search_wellness_knowledge.name}")
RAG tool created: search_wellness_knowledge
Task 8: Building Agentic RAG from Scratch
Now let's put it all together - a complete agentic RAG system built from scratch!

# Define all tools for our RAG agent
rag_tools = [search_wellness_knowledge, calculate, get_current_time]

# Bind tools to the LLM
rag_llm_with_tools = llm.bind_tools(rag_tools)

print("Tools for RAG agent:")
for t in rag_tools:
    print(f"  - {t.name}")
Tools for RAG agent:
  - search_wellness_knowledge
  - calculate
  - get_current_time
# Define the RAG agent components
RAG_SYSTEM_PROMPT = """You are a helpful wellness assistant with access to a comprehensive health and wellness knowledge base.

Your role is to:
1. Answer questions about health, fitness, nutrition, sleep, and mental wellness
2. ALWAYS search the knowledge base when the user asks wellness-related questions
3. Provide accurate, helpful information based on the retrieved context
4. Be supportive and encouraging in your responses
5. If you cannot find relevant information, say so honestly

Remember: Always cite information from the knowledge base when applicable."""

def rag_agent_node(state: AgentState):
    """The RAG agent node - calls the LLM with wellness system prompt."""
    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)] + state["messages"]
    response = rag_llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create tool node for RAG tools
rag_tool_node = ToolNode(rag_tools)

print("RAG agent node defined")
RAG agent node defined
# Build the RAG agent graph
rag_workflow = StateGraph(AgentState)

# Add nodes
rag_workflow.add_node("agent", rag_agent_node)
rag_workflow.add_node("tools", rag_tool_node)

# Set entry point
rag_workflow.add_edge(START, "agent")

# Add conditional edge
rag_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)

# Add edge from tools back to agent
rag_workflow.add_edge("tools", "agent")

# Compile
rag_agent = rag_workflow.compile()

print("Agentic RAG built from scratch!")
Agentic RAG built from scratch!
# Visualize the RAG agent
try:
    from IPython.display import display, Image
    display(Image(rag_agent.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph image: {e}")
    print("\nGraph structure:")
    print(rag_agent.get_graph().draw_ascii())

# Test the RAG agent
print("Testing Agentic RAG (with local models):")
print("=" * 50)

response = rag_agent.invoke({
    "messages": [HumanMessage(content="What are some tips for better sleep?")]
})

print("\nFinal Response:")
print("=" * 50)
print(response["messages"][-1].content)
Testing Agentic RAG (with local models):
==================================================

Final Response:
==================================================
Here are some tips for better sleep:

1. **Establish a consistent sleep schedule**: Go to bed and wake up at the same time every day, including weekends.
2. **Create a relaxing bedtime routine**: Engage in calming activities like reading, gentle stretching, or taking a warm bath before bed.
3. **Optimize your sleep environment**: Keep your bedroom cool (around 65-68°F), dark, and quiet. Consider using blackout curtains, earplugs, or a white noise machine if necessary.
4. **Limit screen exposure before bed**: Avoid screens for at least an hour before bedtime to minimize blue light exposure.
5. **Avoid stimulating activities before bed**: Try to avoid caffeine, heavy meals, and intense exercise within 2-3 hours of bedtime.
6. **Exercise regularly, but not too close to bedtime**: Regular physical activity can help improve sleep quality, but avoid vigorous exercise within a few hours of bedtime.
7. **Consider relaxation techniques**: Techniques like progressive muscle relaxation, meditation, or deep breathing exercises can help calm your mind and body before bed.

Remember that it may take some time to notice improvements in your sleep quality. Be patient, and don't hesitate to consult with a healthcare professional if you continue to struggle with sleep.

Additional resources:

* National Sleep Foundation: [www.sleepfoundation.org](http://www.sleepfoundation.org)
* American Academy of Sleep Medicine: [aasm.org](http://aasm.org)

Please note that these tips are based on general wellness knowledge and may not be applicable to everyone. If you have specific concerns or questions, it's always best to consult with a healthcare professional.
# Test with a complex query requiring both RAG and calculation
print("Testing with complex query:")
print("=" * 50)

response = rag_agent.invoke({
    "messages": [HumanMessage(
        content="I'm stressed and sleeping poorly. What should I do? Also, if I sleep 6 hours a night for a week, how many total hours is that?"
    )]
})

print("\nFinal Response:")
print("=" * 50)
print(response["messages"][-1].content)
Testing with complex query:
==================================================

Final Response:
==================================================
It sounds like you're experiencing stress and poor sleep quality. Here are some tips that may help:

1. **Establish a consistent sleep schedule**: Try to go to bed and wake up at the same time every day, including weekends.
2. **Create a relaxing bedtime routine**: Engage in calming activities like reading, gentle stretching, or taking a warm bath before bed.
3. **Optimize your sleep environment**: Keep your bedroom cool, dark, and quiet. Consider using blackout curtains, earplugs, or a white noise machine if necessary.
4. **Limit screen exposure before bed**: Avoid screens for at least an hour before bedtime, as the blue light they emit can interfere with your body's production of melatonin, a sleep hormone.
5. **Exercise regularly, but not too close to bedtime**: Regular exercise can help improve sleep quality, but avoid vigorous exercise within a few hours of bedtime, as it can actually interfere with sleep.
6. **Try relaxation techniques**: Practice deep breathing, progressive muscle relaxation, or mindfulness meditation to help calm your mind and body before bed.

Remember that it may take some time to notice improvements in your sleep quality. Be patient, and don't hesitate to seek professional help if you continue to struggle with stress and poor sleep.

As for your question about sleeping 6 hours a night for a week, the total number of hours is indeed 42 (6 * 7).
# Test that the agent knows when NOT to use RAG
print("Testing agent decision-making (should NOT use RAG):")
print("=" * 50)

response = rag_agent.invoke({
    "messages": [HumanMessage(content="What is 125 * 8?")]
})

print("\nFinal Response:")
print(response["messages"][-1].content)
Testing agent decision-making (should NOT use RAG):
==================================================

Final Response:
The result of multiplying 125 by 8 is 1000. Is there anything else I can help you with?
❓ Question #3:
Compare the experience of building an agent from scratch with LangGraph versus using create_agent from Session 3. What are the trade-offs between control and convenience? When would you choose one approach over the other?

Answer:
Building an agent from scratch in LangGraph gives you full control: you explicitly define the state, nodes, loop edges, and routing (e.g., should_continue, tool gating, fallbacks, iteration limits). That makes behavior transparent and customizable, and it’s easier to add policies, safety checks, logging, and non-standard workflows.

Using create_agent is more convenient: it packages the common ReAct loop (LLM → tool execution → repeat) into a few lines, so you can prototype quickly. The trade-off is less visibility and fewer insertion points for custom logic unless you use advanced hooks.

When to choose:

From scratch (LangGraph): production-like needs—custom routing, governance/HITL, strict tool control, reliability guardrails, debugging/observability, or multi-step workflows.

create_agent: fast demos/prototypes, standard tool-using agents, and when default ReAct behavior is “good enough.”

❓ Question #4:
We used local models (gpt-oss:20b and embeddinggemma) instead of cloud APIs. What are the advantages and disadvantages of this approach?

Answer:
Using local models via Ollama has clear upsides: privacy and control (data stays on your machine), predictable cost (no per-token API billing), offline development, and the ability to iterate/debug locally with the exact same environment every run (useful for demos and learning).

The trade-offs are capability and reliability: local models are often smaller/weaker than top cloud models, can be less consistent with tool-calling, and are constrained by your hardware (RAM/CPU/VRAM → latency, context limits, and model choice). You also take on ops burden yourself—installing/maintaining Ollama, managing model versions, and no managed scaling/uptime guarantees like cloud APIs.

🏗️ Activity #2: Extend the Agent with Memory
LangGraph supports checkpointing which enables conversation memory across invocations.

Your task: Add memory to the RAG agent so it can:

Remember previous questions in the conversation
Reference past context when answering new questions
Build on previous answers
Hint: Use MemorySaver from langgraph.checkpoint.memory and pass a thread_id in the config.

📚 Documentation:

LangGraph Persistence & Memory
How to add memory to your graph
MemorySaver Reference
### YOUR CODE HERE ###

from langgraph.checkpoint.memory import MemorySaver

# Create a memory saver
memory = MemorySaver()

# Recompile the agent with checkpointing
# rag_agent_with_memory = rag_workflow.compile(checkpointer=memory)

# Test with a conversation that requires memory
# Use config={"configurable": {"thread_id": "conversation-1"}}
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# 1) Create a memory saver (in-memory checkpointer)
memory = MemorySaver()

# 2) Recompile the SAME rag_workflow with checkpointing enabled
rag_agent_with_memory = rag_workflow.compile(checkpointer=memory)

# 3) Use a stable thread_id so state persists across invocations
cfg = {"configurable": {"thread_id": "conversation-1"}}
# Test your memory-enabled agent with a multi-turn conversation
# ---- Multi-turn test ----

# Turn 1
r1 = rag_agent_with_memory.invoke(
    {"messages": [HumanMessage(content="I'm sleeping poorly. What are 3 practical tips I should try tonight?")]},
    config=cfg
)
print("Turn 1:\n", r1["messages"][-1].content, "\n")

# Turn 2 (refers to Turn 1)
r2 = rag_agent_with_memory.invoke(
    {"messages": [HumanMessage(content="Good. Pick the most important one from your last answer and give me a 2-step plan.")]},
    config=cfg
)
print("Turn 2:\n", r2["messages"][-1].content, "\n")

# Turn 3 (checks it still remembers)
r3 = rag_agent_with_memory.invoke(
    {"messages": [HumanMessage(content="Remind me what you recommended first, and why.")]},
    config=cfg
)
print("Turn 3:\n", r3["messages"][-1].content, "\n")
Turn 1:
 Here are three practical tips to help you improve your sleep:

1. **Establish a consistent sleep schedule**: Go to bed and wake up at the same time every day, including weekends. This helps regulate your body's internal clock and can improve the quality of your sleep.
2. **Create a relaxing bedtime routine**: Develop a calming pre-sleep routine, such as reading a book, taking a warm bath, or practicing gentle stretches. This can help signal to your brain that it's time to sleep.
3. **Optimize your sleep environment**: Make your bedroom a sleep haven by keeping it cool (around 65-68°F), dark, and quiet. Consider using blackout curtains, earplugs, or a white noise machine if necessary. Invest in a comfortable mattress and pillows to promote better sleep.

Remember, it may take some time to notice improvements in your sleep quality. Be patient, and don't hesitate to explore additional resources or consult with a healthcare professional if you continue to struggle with sleep. 

Turn 2:
 Based on the knowledge base, I recommend maintaining a consistent sleep schedule as the most important tip. Here's a 2-step plan to help you achieve this:

**Step 1: Set a specific bedtime and wake-up time**
Choose a realistic bedtime and wake-up time that works for your lifestyle, and commit to it even on weekends. For example, if you're not a morning person, don't set your alarm for 6 AM. Aim for 7-9 hours of sleep each night.

**Step 2: Gradually adjust your schedule**
If you're not used to going to bed at the same time every night, it may take some time for your body to adjust. Start by setting your bedtime 15-30 minutes earlier or later than usual and gradually work your way towards your desired schedule over the course of a week or two.

Remember, consistency is key when it comes to sleep hygiene. Stick to your schedule as much as possible, even on weekends or days off, to help regulate your body's internal clock and improve the quality of your sleep. 

Turn 3:
 I recommended "Maintain a consistent sleep schedule, even on weekends" as the most important tip for improving sleep. I chose this because a consistent sleep schedule helps regulate your body's internal clock, which can improve the quality of your sleep. This is especially important because many people tend to sleep in later on weekends, but this can disrupt their sleep patterns and make it harder to fall asleep or stay asleep during the week.

By maintaining a consistent sleep schedule, you can help your body get into a routine and improve the quality of your sleep. This can lead to better rest, improved mood, and increased energy levels. 

Advanced Build: Implement a Human-in-the-Loop (HITL) pattern using LangGraph's interrupt feature
Step 1 : Setup + compile HITL graph (interrupt before tools)

# =========================
# Advanced Build (HITL): Interrupt before tools
# =========================

from typing import Any, Dict, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# Preconditions:
# - rag_workflow exists (built in Task 8)
# - tools node in rag_workflow is named exactly "tools"
# - Agent state includes "messages" with add_messages reducer (as in your notebook)

memory = MemorySaver()

rag_agent_hitl = rag_workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"],   # <-- pause right before tools run
)

print("HITL graph compiled. It will PAUSE before the 'tools' node executes.")
HITL graph compiled. It will PAUSE before the 'tools' node executes.
Step 2: Helper functions (show / approve / reject / modify)

def _last_ai_message(state: Dict[str, Any]) -> AIMessage | None:
    msgs = state.get("messages", [])
    if not msgs:
        return None
    last = msgs[-1]
    return last if isinstance(last, AIMessage) else None

def show_proposed_tool_calls(state: Dict[str, Any]) -> bool:
    """
    Prints tool calls proposed by the model (if any).
    Returns True if tool calls exist (meaning HITL approval is needed).
    """
    last = _last_ai_message(state)
    if last is None:
        print("ℹ️ No AI message found.")
        return False

    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("INTERRUPTED BEFORE TOOLS. Model proposed tool call(s):")
        for i, tc in enumerate(tool_calls, 1):
            print(f"  {i}) name={tc.get('name')} args={tc.get('args', {})}")
        return True

    print("ℹ️ Model did NOT propose tool calls. No approval needed.")
    return False

def reject_all_tools(state: Dict[str, Any]) -> AIMessage:
    """
    Reject: strip all tool calls by replacing last AIMessage with an AIMessage without tool_calls.
    (This forces the graph to end or respond without tools.)
    """
    last = _last_ai_message(state)
    patched = AIMessage(content=(last.content if last else ""))
    return patched

def modify_first_tool_args(state: Dict[str, Any], new_args: Dict[str, Any]) -> AIMessage:
    """
    Modify: replace args of the first tool call. Keeps tool name the same.
    This is the simplest modification pattern for demo.
    """
    last = _last_ai_message(state)
    if last is None or not getattr(last, "tool_calls", None):
        raise ValueError("No tool calls to modify.")

    tool_calls = list(last.tool_calls)
    tool_calls[0] = {**tool_calls[0], "args": new_args}

    patched = AIMessage(content=(last.content or ""), tool_calls=tool_calls)
    return patched
Step 3: Run Turn 1 (this should interrupt before tools)

# Use a fresh thread_id for each demo run
thread_id = "hitl-session-4-demo"
config = {"configurable": {"thread_id": thread_id}}

question = "What are 3 tips for better sleep? Please use the knowledge base."
state1 = rag_agent_hitl.invoke(
    {"messages": [HumanMessage(content=question)]},
    config=config
)

print("\n--- Turn 1 result (should interrupt before tools if tool_calls exist) ---")
needs_approval = show_proposed_tool_calls(state1)
--- Turn 1 result (should interrupt before tools if tool_calls exist) ---
INTERRUPTED BEFORE TOOLS. Model proposed tool call(s):
  1) name=search_wellness_knowledge args={'query': 'better sleep tips'}
Step 4: Human decision (Approve / Reject / Modify) : This is interactive: it will prompt to take action in the notebook.

if needs_approval:
    decision = input("\nType: approve / reject / modify : ").strip().lower()

    if decision == "approve":
        print("Approved tool calls (no change).")

    elif decision == "reject":
        patched = reject_all_tools(state1)
        rag_agent_hitl.update_state(config, {"messages": [patched]})
        print("Rejected tool calls. Tools will NOT run.")

    elif decision == "modify":
        # For RAG, the tool is usually search_wellness_knowledge(query=...)
        new_query = input("Enter a better retrieval query: ").strip()
        patched = modify_first_tool_args(state1, {"query": new_query})
        rag_agent_hitl.update_state(config, {"messages": [patched]})
        print("✏️ Modified tool args for first tool call.")

    else:
        print("Unknown choice. Defaulting to approve.")
else:
    print("No approval step needed because no tool call was proposed.")
Approved tool calls (no change).
Step 5: Resume execution (tools run if approved, then agent finishes)

# Resume from the interrupt point using the SAME thread_id
state2 = rag_agent_hitl.invoke(None, config=config)

print("\n--- After RESUME ---")
print("Final answer:\n")
print(state2["messages"][-1].content)
--- After RESUME ---
Final answer:

Here are three tips for better sleep based on the knowledge base:

1. **Establish a consistent sleep schedule**: Go to bed and wake up at the same time every day, including weekends, to regulate your body's internal clock.
2. **Create a relaxing bedtime routine**: Engage in calming activities like reading, gentle stretching, or taking a warm bath to signal to your brain that it's time to sleep.
3. **Optimize your sleep environment**: Keep your bedroom cool (around 65-68°F), dark, and quiet, and invest in a quality mattress and pillows to promote better sleep.

Remember, consistency and relaxation are key to improving the quality of your sleep!
Summary
In this session, we:

Built agents from scratch using LangGraph's low-level primitives (StateGraph, nodes, edges)
Used local open-source models with Ollama (gpt-oss:20b + embeddinggemma)
Transitioned to LangChain for document loading and text splitting
Created an Agentic RAG system that intelligently decides when to retrieve information
Key Takeaways:
StateGraph gives you full control over agent architecture
Conditional edges enable dynamic routing based on LLM decisions
Local models provide privacy and cost savings, with trade-offs in performance
LangSmith provides crucial visibility regardless of where your models run
What's Next?
Now that you understand the fundamentals, you can:

Add more sophisticated routing logic
Implement human-in-the-loop patterns
Build multi-agent systems
Deploy to production with LangGraph Platform
📚 Further Reading:

LangGraph How-To Guides
Human-in-the-Loop Patterns
Multi-Agent Architectures
LangGraph Platform