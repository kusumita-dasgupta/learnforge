
The Agent Loop: Building Production Agents with LangChain 1.0
In this notebook, we'll explore the foundational concepts of AI agents and learn how to build production-grade agents using LangChain's new create_agent abstraction with middleware support.

Learning Objectives:

Understand what an "agent" is and how the agent loop works
Learn the core constructs of LangChain (Runnables, LCEL)
Master the create_agent function and middleware system
Build an agentic RAG application using Qdrant
Table of Contents:
Breakout Room #1: Introduction to LangChain, LangSmith, and create_agent

Task 1: Dependencies
Task 2: Environment Variables
Task 3: LangChain Core Concepts (Runnables & LCEL)
Task 4: Understanding the Agent Loop
Task 5: Building Your First Agent with create_agent()
Question #1 & Question #2
Activity #1: Create a Custom Tool
Breakout Room #2: Middleware - Agentic RAG with Qdrant

Task 6: Loading & Chunking Documents
Task 7: Setting up Qdrant Vector Database
Task 8: Creating a RAG Tool
Task 9: Introduction to Middleware
Task 10: Building Agentic RAG with Middleware
Question #3 & Question #4
Activity #2: Enhance the Agent
🤝 Breakout Room #1
Introduction to LangChain, LangSmith, and create_agent
Task 1: Dependencies
First, let's ensure we have all the required packages installed. We'll be using:

LangChain 1.0+: The core framework with the new create_agent API
LangChain-OpenAI: OpenAI model integrations
LangSmith: Observability and tracing
Qdrant: Vector database for RAG
tiktoken: Token counting for text splitting
# Run this cell to install dependencies (if not using uv sync)
# !pip install langchain>=1.0.0 langchain-openai langsmith langgraph qdrant-client langchain-qdrant tiktoken nest-asyncio
# Core imports we'll use throughout the notebook
import os
import getpass
from uuid import uuid4

import nest_asyncio
nest_asyncio.apply()  # Required for async operations in Jupyter
Task 2: Environment Variables
We need to set up our API keys for:

OpenAI - For the GPT-5 model
LangSmith - For tracing and observability (optional but recommended)
# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
# Optional: Set up LangSmith for tracing
# This provides powerful debugging and observability for your agents

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE9 - The Agent Loop - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key (press Enter to skip): ") or ""

if not os.environ["LANGCHAIN_API_KEY"]:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled")
else:
    print(f"LangSmith tracing enabled. Project: {os.environ['LANGCHAIN_PROJECT']}")
LangSmith tracing enabled. Project: AIE9 - The Agent Loop - 4fb6b4dd
Task 3: LangChain Core Concepts
Before diving into agents, let's understand the fundamental building blocks of LangChain.

What is a Runnable?
A Runnable is the core abstraction in LangChain - think of it as a standardized component that:

Takes an input
Performs some operation
Returns an output
Every component in LangChain (models, prompts, retrievers, parsers) is a Runnable, which means they all share the same interface:

result = runnable.invoke(input)           # Single input
results = runnable.batch([input1, input2]) # Multiple inputs
for chunk in runnable.stream(input):       # Streaming
    print(chunk)
What is LCEL (LangChain Expression Language)?
LCEL allows you to chain Runnables together using the | (pipe) operator:

chain = prompt | model | output_parser
result = chain.invoke({"query": "Hello!"})
This is similar to Unix pipes - the output of one component becomes the input to the next.

# Let's see LCEL in action with a simple example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create our components (each is a Runnable)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks like a pirate."),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-5", temperature=0.7)

output_parser = StrOutputParser()

# Chain them together with LCEL
pirate_chain = prompt | model | output_parser
# Invoke the chain
response = pirate_chain.invoke({"question": "What is the capital of France?"})
print(response)
Arrr, that’d be Paris, matey!
Task 4: Understanding the Agent Loop
What is an Agent?
An agent is a system that uses an LLM to decide what actions to take. Unlike a simple chain that follows a fixed sequence, an agent can:

Reason about what to do next
Take actions by calling tools
Observe the results
Iterate until the task is complete
The Agent Loop
The core of every agent is the agent loop:

                          AGENT LOOP                         
                                                             
      +----------+     +----------+     +----------+         
      |  Model   | --> |   Tool   | --> |  Model   | --> ... 
      |   Call   |     |   Call   |     |   Call   |         
      +----------+     +----------+     +----------+         
           |                                  |              
           v                                  v              
      "Use search"                   "Here's the answer"     
Model Call: The LLM receives the current state and decides whether to:

Call a tool (continue the loop)
Return a final answer (exit the loop)
Tool Call: If the model decides to use a tool, the tool is executed and its output is added to the conversation

Repeat: The loop continues until the model decides it has enough information to answer

Why create_agent?
LangChain 1.0 introduced create_agent as the new standard way to build agents. It provides:

Simplified API: One function to create production-ready agents
Middleware Support: Hook into any point in the agent loop
Built on LangGraph: Uses the battle-tested LangGraph runtime under the hood
Task 5: Building Your First Agent with create_agent()
Let's build a simple agent that can perform calculations and tell the time.

Step 1: Define Tools
Tools are functions that the agent can call. We use the @tool decorator to create them.

from langchain_core.tools import tool

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

# Create our tool belt
tools = [calculate, get_current_time]

print("Tools created:")
for t in tools:
    print(f"  - {t.name}: {t.description[:60]}...")
Tools created:
  - calculate: Evaluate a mathematical expression. Use this for any math ca...
  - get_current_time: Get the current date and time. Use this when the user asks a...
Step 2: Create the Agent
Now we use create_agent to build our agent. The function takes:

model: The LLM to use (can be a string like "gpt-5" or a model instance)
tools: List of tools the agent can use
prompt: Optional system prompt to customize behavior
from langchain.agents import create_agent

# Create our first agent
simple_agent = create_agent(
    model="gpt-5",
    tools=tools,
    system_prompt="You are a helpful assistant that can perform calculations and tell the time. Always explain your reasoning."
)

print("Agent created successfully!")
print(f"Type: {type(simple_agent)}")
Agent created successfully!
Type: <class 'langgraph.graph.state.CompiledStateGraph'>
Step 3: Run the Agent
The agent is a Runnable, so we can invoke it like any other LangChain component.

# Test the agent with a simple calculation
response = simple_agent.invoke(
    {"messages": [{"role": "user", "content": "What is 25 * 48?"}]}
)

# Print the final response
print("Agent Response:")
print(response["messages"][-1].content)
Agent Response:
25 × 48 = 1200.

Reasoning: 25 × 48 = 25 × (50 − 2) = 25×50 − 25×2 = 1250 − 50 = 1200.
# Test with a multi-step question that requires multiple tool calls
response = simple_agent.invoke(
    {"messages": [{"role": "user", "content": "What time is it, and what is 100 divided by the current hour?"}]}
)

print("Agent Response:")
print(response["messages"][-1].content)
Agent Response:
Here’s how I worked it out:

- Current time: 2026-01-25 20:30:05 (from the system clock).
- The current hour is 20.
- 100 divided by 20 = 5.0.
# Let's see the full conversation to understand the agent loop
print("Full Agent Conversation:")
print("=" * 50)
for msg in response["messages"]:
    role = msg.type if hasattr(msg, 'type') else 'unknown'
    content = msg.content if hasattr(msg, 'content') else str(msg)
    print(f"\n[{role.upper()}]")
    print(content[:500] if len(str(content)) > 500 else content)
Full Agent Conversation:
==================================================

[HUMAN]
What time is it, and what is 100 divided by the current hour?

[AI]


[TOOL]
The current date and time is: 2026-01-25 20:30:05

[AI]


[TOOL]
The result of 100 / 20 is 5.0

[AI]
Here’s how I worked it out:

- Current time: 2026-01-25 20:30:05 (from the system clock).
- The current hour is 20.
- 100 divided by 20 = 5.0.
Streaming Agent Responses
For better UX, we can stream the agent's responses as they're generated.

# Stream the agent's response
print("Streaming Agent Response:")
print("=" * 50)

for chunk in simple_agent.stream(
    {"messages": [{"role": "user", "content": "Calculate 15% of 250"}]},
    stream_mode="updates"
):
    for node, values in chunk.items():
        print(f"\n[Node: {node}]")
        if "messages" in values:
            for msg in values["messages"]:
                if hasattr(msg, 'content') and msg.content:
                    print(msg.content)
Streaming Agent Response:
==================================================

[Node: model]
To find 15% of 250, I convert 15% to a decimal (0.15) and multiply:
0.15 × 250. I'll calculate that now.

[Node: tools]
The result of 0.15 * 250 is 37.5

[Node: model]
I converted 15% to the decimal 0.15 and multiplied 0.15 × 250 = 37.5. So, 15% of 250 is 37.5.
❓ Question #1:
In the agent loop, what determines whether the agent continues to call tools or returns a final answer to the user? How does create_agent handle this decision internally?

✅ Answer:
In the agent loop, the model’s output determines whether the agent keeps going or stops. If the LLM returns a message that includes a tool call (i.e., “call tool X with these args”), the agent continues: it executes the tool, adds the tool result (observation) back into the state/messages, and calls the model again. If the LLM returns a normal final assistant message with no tool calls, the agent stops and that message becomes the final answer.

How create_agent handles it internally: create_agent builds a small control loop (LangGraph under the hood) that repeatedly: i. Calls the model with the current state (messages + any runtime context). ii. Inspects the model response: - Tool calls present? → route to a “tool execution” step. - No tool calls? → route to an “end” step (return the answer). iii. After tool execution, it appends the tool outputs as new messages and goes back to step 1.

So the decision is not a hard-coded rule in your code; it’s a routing decision based on whether the model requested a tool call in its structured output.

❓ Question #2:
Looking at the calculate and get_current_time tools we created, why is the docstring so important for each tool? How does the agent use this information when deciding which tool to call?

✅ Answer:
The docstring is the tool’s interface contract for the LLM. It tells the agent what the tool does, when it should be used, and what inputs it expects. The agent does not read the Python implementation; it only sees the name, description (from the docstring), and argument schema derived from it.

How the agent uses the docstring: When the model is deciding what to do next, it: i. Reads the user request ii. Semantically matches the request against each tool’s docstring and argument descriptions iii. Chooses the tool whose description best fits the task (e.g., math → calculate, time/date → get_current_time) iv. Generates a structured tool call with arguments that match the docstring-defined schema

If the docstring is vague or misleading, the agent may:

Call the wrong tool
Pass incorrect arguments
Not call any tool at all
In short, clear, specific docstrings are what make tools discoverable and usable by the agent.

🏗️ Activity #1: Create a Custom Tool
Create your own custom tool and add it to the agent!

Ideas:

A tool that converts temperatures between Celsius and Fahrenheit
A tool that generates a random number within a range
A tool that counts words in a given text
Requirements:

Use the @tool decorator
Include a clear docstring (this is what the agent sees!)
Add it to the agent and test it
Step 1: Create the custom tool
### YOUR CODE HERE ###

# Create your custom tool
# @tool
# def my_custom_tool():
#     """Your tool description here - be clear about what it does!"""
#     pass
from langchain_core.tools import tool
import re

@tool
def analyze_text_complexity(text: str) -> str:
    """Analyze the complexity of a given text and provide basic readability signals.
    
    Use this tool when you need to:
    - Assess how complex or dense a piece of text is
    - Get quick readability signals for documentation, prompts, or user content
    - Compare writing styles (simple vs complex)
    
    Args:
        text: The text to analyze
    """
    if not text.strip():
        return "No text provided for analysis."

    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)

    word_count = len(words)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))

    avg_words_per_sentence = round(word_count / sentence_count, 2)

    if avg_words_per_sentence < 12:
        complexity = "Simple"
    elif avg_words_per_sentence < 20:
        complexity = "Moderate"
    else:
        complexity = "Complex"

    return (
        f"Text Analysis:\n"
        f"- Words: {word_count}\n"
        f"- Sentences: {sentence_count}\n"
        f"- Avg words/sentence: {avg_words_per_sentence}\n"
        f"- Estimated complexity: {complexity}"
    )
Step 2: Extend your existing tool list
# Add your tool to the tools list 
tools = [
    calculate,
    get_current_time,
    analyze_text_complexity
]
Step 3: Create the agent
simple_agent = create_agent(
    model="gpt-5",
    tools=tools,
    system_prompt=(
        "You are a helpful assistant that can perform calculations, "
        "analyze text, and tell the time. "
        "Always explain your reasoning."
    )
)
Step 4: Test the tool via the agent (agent decides)
# Test your custom tool with the agent
response = simple_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze the complexity of this text: "
                           "LangChain introduces a composable abstraction "
                           "for building agentic systems using language models."
            }
        ]
    }
)

print(response["messages"][-1].content)
Here’s a quick complexity readout and why:

- Basic metrics (from analysis tool): 12 words, 1 sentence, average 12 words per sentence → structurally simple.
- Vocabulary/jargon: Contains specialized terms (“composable abstraction,” “agentic systems,” “language models”). These raise conceptual difficulty despite the short sentence.
- Overall: Moderate complexity. Syntax is easy; terminology makes it harder for non-experts. Suitable for a technical audience; may challenge general readers.

If you want a simpler version:
- “LangChain provides building blocks for creating AI agents that use language models.”
🤝 Breakout Room #2
Middleware - Agentic RAG with Qdrant
Now that we understand the basics of agents, let's build something more powerful: an Agentic RAG system.

Traditional RAG follows a fixed pattern: retrieve → generate. But Agentic RAG gives the agent control over when and how to retrieve information, making it more flexible and intelligent.

We'll also introduce middleware - hooks that let us customize the agent's behavior at every step.

Task 6: Loading & Chunking Documents
We'll use the same Health & Wellness Guide from Session 2 to maintain continuity.

# Load the document using our aimakerspace utilities
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter

# Load the document
text_loader = TextFileLoader("data/HealthWellnessGuide.txt")
documents = text_loader.load_documents()

print(f"Loaded {len(documents)} document(s)")
print(f"Total characters: {sum(len(doc) for doc in documents):,}")
Loaded 1 document(s)
Total characters: 16,206
# Split the documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_texts(documents)

print(f"Split into {len(chunks)} chunks")
print(f"\nSample chunk:")
print("-" * 50)
print(chunks[0][:300] + "...")
Split into 41 chunks

Sample chunk:
--------------------------------------------------
The Personal Wellness Guide
A Comprehensive Resource for Health and Well-being

PART 1: EXERCISE AND MOVEMENT

Chapter 1: Understanding Exercise Basics

Exercise is one of the most important things you can do for your health. Regular physical activity can improve your brain health, help manage weigh...
Task 7: Setting up Qdrant Vector Database
Qdrant is a production-ready vector database. We'll use an in-memory instance for development, but the same code works with a hosted Qdrant instance.

Key concepts:

Collection: A namespace for storing vectors (like a table in SQL)
Points: Individual vectors with optional payloads (metadata)
Distance: How similarity is measured (we'll use cosine similarity)
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Get embedding dimension
sample_embedding = embedding_model.embed_query("test")
embedding_dim = len(sample_embedding)
print(f"Embedding dimension: {embedding_dim}")
Embedding dimension: 1536
# Create Qdrant client (in-memory for development)
qdrant_client = QdrantClient(":memory:")

# Create a collection for our wellness documents
collection_name = "wellness_knowledge_base"

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embedding_dim,
        distance=Distance.COSINE
    )
)

print(f"Created collection: {collection_name}")
Created collection: wellness_knowledge_base
# Create the vector store and add documents
from langchain_core.documents import Document

# Convert chunks to LangChain Document objects
langchain_docs = [Document(page_content=chunk) for chunk in chunks]

# Create vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model
)

# Add documents to the vector store
vector_store.add_documents(langchain_docs)

print(f"Added {len(langchain_docs)} documents to vector store")
Added 41 documents to vector store
# Test the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

test_results = retriever.invoke("How can I improve my sleep?")

print("Retrieved documents:")
for i, doc in enumerate(test_results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:200] + "...")
Retrieved documents:

--- Document 1 ---
 memory and learning

Chapter 8: Improving Sleep Quality

Sleep hygiene refers to habits and practices that promote consistent, quality sleep.

Essential sleep hygiene practices:
- Maintain a consiste...

--- Document 2 ---
 Avoid caffeine after 2 PM
- Exercise regularly, but not too close to bedtime
- Limit alcohol and heavy meals before bed

Creating an optimal sleep environment:
- Temperature: 65-68 degrees Fahrenheit...

--- Document 3 ---
de for sunlight
4. Power pose for 2 minutes
5. Healthy snack (nuts, fruit)
6. Brief walk around the block
7. Upbeat music
8. Splash cold water on face

Sleep Checklist:
- Room temperature 65-68F
- Bla...
Task 8: Creating a RAG Tool
Now we'll wrap our retriever as a tool that the agent can use. This is the key to Agentic RAG - the agent decides when to retrieve information.

from langchain_core.tools import tool

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

print(f"Tool created: {search_wellness_knowledge.name}")
print(f"Description: {search_wellness_knowledge.description[:100]}...")
Tool created: search_wellness_knowledge
Description: Search the wellness knowledge base for information about health, fitness, nutrition, sleep, and ment...
Task 9: Introduction to Middleware
Middleware in LangChain 1.0 allows you to hook into the agent loop at various points:

                       MIDDLEWARE HOOKS                 
                                                        
   +--------------+                    +--------------+ 
   | before_model | --> MODEL CALL --> | after_model  | 
   +--------------+                    +--------------+ 
                                                        
   +-------------------+                                
   | wrap_model_call   |  (intercept and modify calls)  
   +-------------------+                                
Common use cases:

Logging: Track what the agent is doing
Guardrails: Filter or modify inputs/outputs
Rate limiting: Control API usage
Human-in-the-loop: Pause for human approval
LangChain provides middleware through decorator functions that hook into specific points in the agent loop.

from langchain.agents.middleware import before_model, after_model

# Track how many model calls we've made
model_call_count = 0

@before_model
def log_before_model(state, runtime):
    """Called before each model invocation."""
    global model_call_count
    model_call_count += 1
    message_count = len(state.get("messages", []))
    print(f"[LOG] Model call #{model_call_count} - Messages in state: {message_count}")
    return None  # Return None to continue without modification

@after_model
def log_after_model(state, runtime):
    """Called after each model invocation."""
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    if last_message:
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        print(f"[LOG] After model - Tool calls requested: {has_tool_calls}")
    return None

print("Logging middleware created!")
Logging middleware created!
# You can also use the built-in ModelCallLimitMiddleware to prevent runaway agents
from langchain.agents.middleware import ModelCallLimitMiddleware

# This middleware will stop the agent after 10 model calls per thread
call_limiter = ModelCallLimitMiddleware(
    thread_limit=10,  # Max calls per conversation thread
    run_limit=5,      # Max calls per single run
    exit_behavior="end"  # What to do when limit is reached
)

print("Call limit middleware created!")
print(f"  - Thread limit: {call_limiter.thread_limit}")
print(f"  - Run limit: {call_limiter.run_limit}")
Call limit middleware created!
  - Thread limit: 10
  - Run limit: 5
Task 10: Building Agentic RAG with Middleware
Now let's put it all together: an agentic RAG system with middleware support!

from langchain.agents import create_agent

# Reset the call counter
model_call_count = 0

# Define our tools - include the RAG tool and the calculator from earlier
rag_tools = [
    search_wellness_knowledge,
    calculate,
    get_current_time
]

# Create the agentic RAG system with middleware
wellness_agent = create_agent(
    model="gpt-5",
    tools=rag_tools,
    system_prompt="""You are a helpful wellness assistant with access to a comprehensive health and wellness knowledge base.

Your role is to:
1. Answer questions about health, fitness, nutrition, sleep, and mental wellness
2. Always search the knowledge base when the user asks wellness-related questions
3. Provide accurate, helpful information based on the retrieved context
4. Be supportive and encouraging in your responses
5. If you cannot find relevant information, say so honestly

Remember: Always cite information from the knowledge base when applicable.""",
    middleware=[
        log_before_model,
        log_after_model,
        call_limiter
    ]
)

print("Wellness Agent created with middleware!")
Wellness Agent created with middleware!
# Test the wellness agent
print("Testing Wellness Agent")
print("=" * 50)

response = wellness_agent.invoke(
    {"messages": [{"role": "user", "content": "What are some tips for better sleep?"}]}
)

print("\n" + "=" * 50)
print("FINAL RESPONSE:")
print("=" * 50)
print(response["messages"][-1].content)
Testing Wellness Agent
==================================================
[LOG] Model call #1 - Messages in state: 1
[LOG] After model - Tool calls requested: [{'name': 'search_wellness_knowledge', 'args': {'query': 'tips for better sleep, sleep hygiene, improving sleep quality, bedtime routine, circadian rhythm, caffeine, screens, bedroom environment, insomnia strategies'}, 'id': 'call_IDmuwnDDq9MJQlb1ap8THeYV', 'type': 'tool_call'}]
[LOG] Model call #2 - Messages in state: 3
[LOG] After model - Tool calls requested: []

==================================================
FINAL RESPONSE:
==================================================
Here are evidence-based tips to improve sleep quality:

- Keep a consistent sleep/wake schedule every day, including weekends [Source 1].
- Create a relaxing wind-down routine (e.g., reading, gentle stretching, warm bath) before bed [Source 1].
- Make your bedroom sleep-friendly: cool, dark, and quiet. Aim for 65–68°F (18–20°C), use blackout curtains or a sleep mask, and consider white noise or earplugs [Source 1; Source 2].
- Limit screens 1–2 hours before bed to reduce blue-light stimulation [Source 1].
- Avoid caffeine after 2 PM, and limit alcohol and heavy meals close to bedtime [Source 1; Source 2].
- Exercise regularly, but try to finish workouts several hours before bedtime [Source 1; Source 2].
- Consider relaxation techniques such as progressive muscle relaxation; if insomnia persists, Cognitive Behavioral Therapy for Insomnia (CBT-I) is an effective, first-line approach [Source 3].

If your sleep problems occur at least 3 nights a week for 3 months or more, that’s considered chronic insomnia—CBT-I can help [Source 3].

Would you like me to tailor these tips to your routine (e.g., trouble falling asleep vs. staying asleep)?
# Test with a more complex query
print("Testing with complex query")
print("=" * 50)

response = wellness_agent.invoke(
    {"messages": [{"role": "user", "content": "I'm feeling stressed and having trouble sleeping. What should I do, and if I sleep 6 hours a night for a week, how many total hours is that?"}]}
)
print("\n" + "=" * 50)
print("FINAL RESPONSE:")
print("=" * 50)
print(response["messages"][-1].content)
Testing with complex query
==================================================
[LOG] Model call #3 - Messages in state: 1
[LOG] After model - Tool calls requested: [{'name': 'search_wellness_knowledge', 'args': {'query': 'stress management techniques and sleep hygiene tips for trouble sleeping, relaxation methods, CBT-I, caffeine, screen time, bedtime routine, breathing exercises, when to seek help'}, 'id': 'call_4akOpg153eZBsufMPvx9loel', 'type': 'tool_call'}, {'name': 'calculate', 'args': {'expression': '6 * 7'}, 'id': 'call_tZjcbTqnnO8NKFES5BIj2RpM', 'type': 'tool_call'}]
[LOG] Model call #4 - Messages in state: 4
[LOG] After model - Tool calls requested: []

==================================================
FINAL RESPONSE:
==================================================
I’m sorry you’re going through this—stress and poor sleep can feed into each other. Here are evidence-based steps you can try, plus your math answer at the end.

Immediate stress relief (you can do these today) [Source 2]:
- Deep breathing (box breathing): inhale 4 counts, hold 4, exhale 4.
- Progressive muscle relaxation: gently tense and release each muscle group from toes to head.
- 5-4-3-2-1 grounding: name 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste.
- Take a short walk (nature if possible) or listen to calming music.

Better sleep strategies [Sources 1, 3]:
- Try relaxation before bed: progressive muscle relaxation, meditation, and deep breathing.
- Consider CBT-I (Cognitive Behavioral Therapy for Insomnia)—a first-line, effective approach for falling and staying asleep.
- Some people find herbal teas like chamomile or valerian helpful; magnesium may help but check with your healthcare provider first.

When to seek extra help [Source 3]:
- If sleep trouble occurs at least 3 nights/week for 3 months or more (chronic insomnia), consider talking with a healthcare provider or a CBT-I specialist.

Your math: 6 hours/night for 7 nights = 42 hours total.

If you’d like, tell me a bit about your current evening routine and what tends to keep you up (mind racing, waking often, etc.), and I can tailor a simple wind-down plan for you tonight.
# Test the agent's ability to know when NOT to use RAG
print("Testing agent decision-making (should NOT use RAG)")
print("=" * 50)

response = wellness_agent.invoke(
    {"messages": [{"role": "user", "content": "What is 125 * 8?"}]}
)

print("\n" + "=" * 50)
print("FINAL RESPONSE:")
print("=" * 50)
print(response["messages"][-1].content)
Testing agent decision-making (should NOT use RAG)
==================================================
[LOG] Model call #3 - Messages in state: 1
[LOG] After model - Tool calls requested: [{'name': 'calculate', 'args': {'expression': '125 * 8'}, 'id': 'call_S1rSXyULqkBJZw6962J0H8EM', 'type': 'tool_call'}]
[LOG] Model call #4 - Messages in state: 3
[LOG] After model - Tool calls requested: []

==================================================
FINAL RESPONSE:
==================================================
125 * 8 = 1000
Visualizing the Agent
The agent created by create_agent is built on LangGraph, so we can visualize its structure.

# Display the agent graph
try:
    from IPython.display import display, Image
    display(Image(wellness_agent.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph: {e}")
    print("\nAgent structure:")
    print(wellness_agent.get_graph().draw_ascii())

❓ Question #3:
How does Agentic RAG differ from traditional RAG? What are the advantages and potential disadvantages of letting the agent decide when to retrieve information?

✅ Answer:
Traditional RAG follows a fixed, linear pipeline: retrieve first, then generate. Every user query triggers retrieval, whether or not external knowledge is actually needed. This makes the system simple and predictable, but also rigid—unnecessary retrievals can add latency, cost, and noise.

Agentic RAG, in contrast, puts an agent (LLM) in control of retrieval. The agent reasons about the user’s question and decides if, when, and how often to retrieve information as part of its loop. Retrieval becomes a tool the agent can invoke selectively, alongside other tools (e.g., calculators, time, APIs).

Advantages of Agentic RAG i. Efficiency: Retrieval is only performed when needed, reducing latency and token cost. ii. Better relevance: The agent can refine or re-run retrieval based on intermediate reasoning. iii. Tool orchestration: Retrieval can be combined with other tools in multi-step tasks. iv. More natural behavior: Mirrors how humans solve problems (think → look up → think again).

Potential disadvantages: i. Less predictability: The agent might skip retrieval when it would have helped, or retrieve too much. ii. More complexity: Harder to debug than a fixed retrieve-then-generate pipeline. iii. Prompt sensitivity: Poor tool descriptions or system prompts can lead to suboptimal retrieval decisions. iv. Control challenges: Requires guardrails (e.g., middleware, call limits) in production to avoid misuse.

In short, Agentic RAG trades simplicity and determinism for flexibility and intelligence, which is powerful but requires stronger observability and safeguards in production systems.

❓ Question #4:
Looking at the middleware examples (log_before_model, log_after_model, and ModelCallLimitMiddleware), describe a real-world scenario where middleware would be essential for a production agent. What specific middleware hooks would you use and why?

✅ Answer:
A real-world scenario where middleware is essential is a production internal developer agent (e.g., “CI/CD helper” or “code cleanup + unit test generator” agent) that can call tools like Jenkins, Git, test runners, and code-modification utilities. In production, you must control cost, safety, and runaway loops, and you need auditability for every model/tool action. Middleware is the mechanism that enforces these guarantees consistently.

What middleware hooks I would use and why:

before_model (like log_before_model) Why: Capture the inputs/state going into each model call (message count, tool outputs so far, user identity/tenant, request type). What it enables: i. Structured logging + trace correlation IDs ii. Guardrails: redact secrets before they reach the model (tokens, credentials, keys) iii. Policy checks: “Is this user allowed to trigger deploy tools?”

after_model (like log_after_model) Why: Inspect the model’s decision before executing it. What it enables: i. Detect whether it requested a tool call, and which tool ii. Block or rewrite unsafe tool calls (e.g., “delete branch main”, “deploy to prod”) iii. Add human-in-the-loop approval for sensitive operations

wrap_model_call (intercept and modify the call) Why: Enforce runtime controls around the model call itself. What it enables: i. Rate limiting / retries / backoff ii. Injecting system constraints (“do not modify code unless tests pass”) iii. Streaming transformations or output filtering

ModelCallLimitMiddleware Why: Prevent runaway agents and unexpected costs. What it enables: i. Hard stop after N model calls per run/thread ii. Predictable cost ceilings iii. Protection against infinite loops when tools return ambiguous results

Why middleware is “essential” here: Without middleware, the agent can: i. Loop indefinitely (cost + reliability risk) ii. Call powerful tools without guardrails (security risk) iii. Be impossible to debug (no consistent logs/traces) iv. Accidentally leak secrets into prompts (compliance risk)

So in production, middleware becomes the control plane for agent behavior: observability + safety + cost management.

🏗️ Activity #2: Enhance the Agentic RAG System
Now it's your turn! Enhance the wellness agent by implementing ONE of the following:

Option A: Add a New Tool
Create a new tool that the agent can use. Ideas:

A tool that calculates BMI given height and weight
A tool that estimates daily calorie needs
A tool that creates a simple workout plan
Option B: Create Custom Middleware
Build middleware that adds new functionality:

Middleware that tracks which tools are used most frequently
Middleware that adds a friendly greeting to responses
Middleware that enforces a response length limit
Option C: Improve the RAG Tool
Enhance the retrieval tool:

Add metadata filtering
Implement reranking of results
Add source citations with relevance scores
Add Middleware to track tool usage
Step 1: Add to the existing middleware definitions

### YOUR CODE HERE ###

# Implement your enhancement below
from collections import Counter
from langchain.agents.middleware import after_model

# Global counters (persist across runs)
tool_usage_total = Counter()

# Per-run counters (reset each run)
tool_usage_run = Counter()

@after_model
def track_tool_usage(state, runtime):
    """Track which tools the model requests."""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name:
            tool_usage_run[name] += 1
            tool_usage_total[name] += 1

    return None
Why after_model? Because that’s when the model decides tool calls and exposes them via tool_calls. It’s the cleanest place to count tool invocations.

Setp 2: Add a small helper to print a summary

def print_tool_usage_summary():
    if not tool_usage_run:
        print("No tools were used in this run.")
        return

    print("\nTool usage (this run):")
    for name, count in tool_usage_run.most_common():
        print(f"  - {name}: {count}")

    print("\nTool usage (total across notebook):")
    for name, count in tool_usage_total.most_common():
        print(f"  - {name}: {count}")
Step 3: Reset per-run counter before each invoke

tool_usage_run.clear()
Step 4: Attach middleware to your wellness agent

wellness_agent = create_agent(
    model="gpt-5",
    tools=rag_tools,
    system_prompt="""You are a helpful wellness assistant with access to a comprehensive health and wellness knowledge base.

Your role is to:
1. Answer questions about health, fitness, nutrition, sleep, and mental wellness
2. Always search the knowledge base when the user asks wellness-related questions
3. Provide accurate, helpful information based on the retrieved context
4. Be supportive and encouraging in your responses
5. If you cannot find relevant information, say so honestly

Important:
- Do NOT use the knowledge base tool for math/time/general non-wellness questions.
- Use calculate for math and get_current_time for time/date.
- When you use the knowledge base, base your answer on it and cite sources in your response.""",
    middleware=[
        log_before_model,
        log_after_model,
        track_tool_usage,
        call_limiter
    ]
)
Step 5: Test the enhanced Agent

# Test your enhanced agent here
tool_usage_run.clear()
response = wellness_agent.invoke(
    {"messages": [{"role": "user", "content": "What are some tips for better sleep?"}]}
)
print(response["messages"][-1].content)
print_tool_usage_summary()
[LOG] Model call #7 - Messages in state: 1
[LOG] After model - Tool calls requested: [{'name': 'search_wellness_knowledge', 'args': {'query': 'evidence-based tips for better sleep, sleep hygiene, improving sleep quality, insomnia self-care'}, 'id': 'call_Ddg0Y2DpQCgEuyR7Y8bPUZFX', 'type': 'tool_call'}]
[LOG] Model call #8 - Messages in state: 3
[LOG] After model - Tool calls requested: []
Here are evidence-based tips to improve your sleep:

- Keep a consistent sleep and wake time every day, including weekends (Source 1; Source 3)
- Create a 30–60 minute wind-down routine (reading, gentle stretching, warm bath) before bed (Source 1)
- Optimize your sleep environment: cool (about 65–68°F), dark, and quiet; consider blackout curtains or a sleep mask; use a comfortable mattress and pillows (Source 1; Source 3)
- Avoid screens for 1–2 hours before bedtime (Source 1; Source 3)
- Cut off caffeine by early afternoon (around 2 PM) (Source 1; Source 3)
- Exercise regularly, but finish vigorous workouts several hours before bed (Source 1)
- Go easy on evening alcohol, which can disrupt sleep quality (Source 1)
- Try relaxation techniques such as progressive muscle relaxation or deep breathing as part of your wind-down (Source 2)
- If insomnia (trouble falling asleep, staying asleep, or early waking) persists 3+ nights/week for 3 months, consider Cognitive Behavioral Therapy for Insomnia (CBT-I), a first-line, effective treatment; soothing herbal teas may also help some people (Source 2)

If you’d like, tell me a bit about your current routine and I can help tailor these tips to you.

Sources:
- Source 1: Chapter 8: Improving Sleep Quality (sleep hygiene basics)
- Source 2: Managing Insomnia (CBT-I and relaxation)
- Source 3: Sleep Checklist (environment and schedule specifics)

Tool usage (this run):
  - search_wellness_knowledge: 1

Tool usage (total across notebook):
  - search_wellness_knowledge: 2