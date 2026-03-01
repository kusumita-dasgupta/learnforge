
Multi-Agent Applications: Building Agent Teams with LangGraph
In this notebook, we'll explore multi-agent systems - applications where multiple specialized agents collaborate to solve complex tasks. We'll build on our LangGraph foundation from Session 4 and create agent teams for our Personal Wellness Assistant.

Learning Objectives:

Understand when and why to use multi-agent systems
Master the Supervisor pattern for orchestrating agent teams
Implement Agent Handoffs for dynamic task routing
Use Tavily Search for web research capabilities
Apply context engineering principles to optimize agent performance
Visualize and debug multi-agent systems with LangSmith
Table of Contents:
Breakout Room #1: Multi-Agent Fundamentals & Supervisor Pattern

Task 1: Dependencies & Environment Setup
Task 2: Understanding Multi-Agent Systems
Task 3: Building a Supervisor Agent Pattern
Task 4: Adding Tavily Search for Web Research
Question #1 & Question #2
Activity #1: Add a Custom Specialist Agent
Breakout Room #2: Handoffs & Context Engineering

Task 5: Agent Handoffs Pattern
Task 6: Building a Wellness Agent Team
Task 7: Context Engineering & Optimization
Task 8: Visualizing and Debugging with LangSmith
Question #3 & Question #4
Activity #2: Implement Hierarchical Teams
🤝 Breakout Room #1
Multi-Agent Fundamentals & Supervisor Pattern
Task 1: Dependencies & Environment Setup
Before we begin, make sure you have:

API Keys for:

OpenAI (for GPT-5.2 supervisor and GPT-4o-mini specialist agents)
Tavily (free tier at tavily.com)
LangSmith (optional, for tracing)
Dependencies installed via uv sync

Models Used:

GPT-5.2: Supervisor/orchestrator agents (better reasoning for routing decisions)
GPT-4o-mini: Specialist agents (cost-effective for domain-specific tasks)
Documentation:

Tavily Search API
# Core imports
import os
import getpass
import json
from uuid import uuid4
from typing import Annotated, TypedDict, Literal, Sequence
import operator

import nest_asyncio
nest_asyncio.apply()  # Required for async operations in Jupyter
# Set API Keys
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
# Tavily API Key for web search
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key: ")
# Optional: LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE9 - Multi-Agent Applications - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key (press Enter to skip): ") or ""

if not os.environ["LANGCHAIN_API_KEY"]:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled")
else:
    print(f"LangSmith tracing enabled. Project: {os.environ['LANGCHAIN_PROJECT']}")
LangSmith tracing enabled. Project: AIE9 - Multi-Agent Applications - 7e331e49
# Initialize LLMs - GPT-5.2 for supervisors, GPT-4o-mini for specialists
from langchain_openai import ChatOpenAI

# Supervisor model - better reasoning for routing and orchestration
supervisor_llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# Specialist model - cost-effective for domain-specific tasks
specialist_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Test both models
print("Testing models...")
supervisor_response = supervisor_llm.invoke("Say 'Supervisor ready!' in exactly 2 words.")
specialist_response = specialist_llm.invoke("Say 'Specialist ready!' in exactly 2 words.")

print(f"Supervisor (GPT-5.2): {supervisor_response.content}")
print(f"Specialist (GPT-4o-mini): {specialist_response.content}")
Testing models...
Supervisor (GPT-5.2): Supervisor ready!
Specialist (GPT-4o-mini): Specialist ready!
Task 2: Understanding Multi-Agent Systems
When to Use Multi-Agent Systems
Before building multi-agent systems, ask yourself:

"Do I really need several specialized dynamic reasoning machines collaborating to solve this task more effectively than a single agent could?"

Multi-agent systems are useful when:

Tool/responsibility grouping: Different tasks require different expertise
Prompt separation: Different agents need different instructions/few-shot examples
Piecewise optimization: Easier to improve individual components
Key Multi-Agent Patterns
Pattern	Description	Use Case
Supervisor	Central orchestrator routes to specialist agents	Task delegation, quality control
Handoffs	Agents transfer control to each other	Conversation flows, expertise routing
Hierarchical	Supervisors manage teams of agents	Large-scale systems, departments
Network/Swarm	Agents communicate freely	Collaborative problem-solving
Context Engineering Principles
From leading practitioners:

Dex Horthy (12-Factor Agents): "Own your context window and treat it like prime real estate"
swyx (Agent Engineering): "Agent reliability = great context construction"
Chroma (Context Rot): "Longer ≠ better when it comes to context"
Documentation:

Building Effective Agents (Anthropic)
Don't Build Multi-Agents (Cognition)
12-Factor Agents
Task 3: Building a Supervisor Agent Pattern
The Supervisor Pattern uses a central agent to:

Analyze incoming requests
Route to the appropriate specialist agent
Aggregate and refine responses
                    ┌─────────────────┐
                    │   Supervisor    │
                    │   (Orchestrator)│
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │  Exercise  │    │  Nutrition │    │   Sleep    │
    │   Agent    │    │   Agent    │    │   Agent    │
    └────────────┘    └────────────┘    └────────────┘
Documentation:

LangGraph Supervisor Tutorial
# Import LangGraph and LangChain components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent  # LangChain 1.0 API
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

print("LangGraph and LangChain components imported!")
LangGraph and LangChain components imported!
# First, let's set up our RAG system for the wellness knowledge base
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load and chunk the wellness document
loader = TextLoader("data/HealthWellnessGuide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded and split into {len(chunks)} chunks")
Loaded and split into 45 chunks
# Set up vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_dim = len(embedding_model.embed_query("test"))

qdrant_client = QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="wellness_multiagent",
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="wellness_multiagent",
    embedding=embedding_model
)
vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"Vector store ready with {len(chunks)} documents")
Vector store ready with 45 documents
# Create specialized tools for each agent domain

@tool
def search_exercise_info(query: str) -> str:
    """Search for exercise, fitness, and workout information from the wellness knowledge base.
    Use this for questions about physical activity, workout routines, and exercise techniques.
    """
    results = retriever.invoke(f"exercise fitness workout {query}")
    if not results:
        return "No exercise information found."
    return "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])

@tool
def search_nutrition_info(query: str) -> str:
    """Search for nutrition, diet, and healthy eating information from the wellness knowledge base.
    Use this for questions about food, meal planning, and dietary guidelines.
    """
    results = retriever.invoke(f"nutrition diet food meal {query}")
    if not results:
        return "No nutrition information found."
    return "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])

@tool
def search_sleep_info(query: str) -> str:
    """Search for sleep, rest, and recovery information from the wellness knowledge base.
    Use this for questions about sleep quality, insomnia, and sleep hygiene.
    """
    results = retriever.invoke(f"sleep rest recovery insomnia {query}")
    if not results:
        return "No sleep information found."
    return "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])

@tool
def search_stress_info(query: str) -> str:
    """Search for stress management and mental wellness information from the wellness knowledge base.
    Use this for questions about stress, anxiety, mindfulness, and mental health.
    """
    results = retriever.invoke(f"stress mental wellness mindfulness anxiety {query}")
    if not results:
        return "No stress management information found."
    return "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])

print("Specialist tools created!")
Specialist tools created!
# Create specialist agents using create_agent (LangChain 1.0 API)
# Each specialist uses GPT-4o-mini for cost efficiency

exercise_agent = create_agent(
    model=specialist_llm,
    tools=[search_exercise_info],
    system_prompt="You are an Exercise Specialist. Help users with workout routines, fitness tips, and physical activity guidance. Always search the knowledge base before answering. Be concise and helpful."
)

nutrition_agent = create_agent(
    model=specialist_llm,
    tools=[search_nutrition_info],
    system_prompt="You are a Nutrition Specialist. Help users with diet advice, meal planning, and healthy eating. Always search the knowledge base before answering. Be concise and helpful."
)

sleep_agent = create_agent(
    model=specialist_llm,
    tools=[search_sleep_info],
    system_prompt="You are a Sleep Specialist. Help users with sleep quality, insomnia, and rest optimization. Always search the knowledge base before answering. Be concise and helpful."
)

stress_agent = create_agent(
    model=specialist_llm,
    tools=[search_stress_info],
    system_prompt="You are a Stress Management Specialist. Help users with stress relief, mindfulness, and mental wellness. Always search the knowledge base before answering. Be concise and helpful."
)

print("Specialist agents created (using GPT-4o-mini with create_agent)!")
Specialist agents created (using GPT-4o-mini with create_agent)!
# Define the supervisor state and routing
from typing import List
from pydantic import BaseModel

# Define routing options - supervisor picks ONE specialist, then that specialist responds
class RouterOutput(BaseModel):
    """The supervisor's routing decision."""
    next: Literal["exercise", "nutrition", "sleep", "stress"]
    reasoning: str

class SupervisorState(TypedDict):
    """State for the supervisor multi-agent system."""
    messages: Annotated[list[BaseMessage], add_messages]
    next: str

print("Supervisor state defined!")
Supervisor state defined!
# Create the supervisor node (using GPT-5.2 for routing decisions)
from langchain_core.prompts import ChatPromptTemplate

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Wellness Supervisor coordinating a team of specialist agents.

Your team:
- exercise: Handles fitness, workouts, physical activity, movement questions
- nutrition: Handles diet, meal planning, healthy eating, food questions
- sleep: Handles sleep quality, insomnia, rest, recovery questions
- stress: Handles stress management, mindfulness, mental wellness, anxiety questions

Based on the user's question, decide which ONE specialist should respond.
Choose the most relevant specialist for the primary topic of the question."""),
    ("human", "User question: {question}\n\nWhich specialist should handle this?")
])

# Create structured output for routing (using GPT-5.2)
routing_llm = supervisor_llm.with_structured_output(RouterOutput)

def supervisor_node(state: SupervisorState):
    """The supervisor decides which agent to route to."""
    # Get the user's question from the last human message
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    # Get routing decision
    prompt_value = supervisor_prompt.invoke({"question": user_question})
    result = routing_llm.invoke(prompt_value)
    
    print(f"[Supervisor GPT-5.2] Routing to: {result.next}")
    print(f"  Reason: {result.reasoning}")
    
    return {"next": result.next}

print("Supervisor node created (using GPT-5.2)!")
Supervisor node created (using GPT-5.2)!
# Create agent nodes that wrap the specialist agents

def create_agent_node(agent, name: str):
    """Create a node that runs a specialist agent and returns the final response."""
    def agent_node(state: SupervisorState):
        print(f"[{name.upper()} Agent] Processing request...")
        
        # Invoke the specialist agent with the conversation
        result = agent.invoke({"messages": state["messages"]})
        
        # Get the agent's final response
        agent_response = result["messages"][-1]
        
        # Add agent identifier to the response
        response_with_name = AIMessage(
            content=f"[{name.upper()} SPECIALIST]\n\n{agent_response.content}",
            name=name
        )
        
        print(f"[{name.upper()} Agent] Response complete.")
        return {"messages": [response_with_name]}
    
    return agent_node

# Create nodes for each specialist
exercise_node = create_agent_node(exercise_agent, "exercise")
nutrition_node = create_agent_node(nutrition_agent, "nutrition")
sleep_node = create_agent_node(sleep_agent, "sleep")
stress_node = create_agent_node(stress_agent, "stress")

print("Agent nodes created!")
Agent nodes created!
# Build the supervisor graph
# KEY: Specialists go directly to END (no loop back to supervisor)

def route_to_agent(state: SupervisorState) -> str:
    """Route to the next agent based on supervisor decision."""
    return state["next"]

# Create the graph
supervisor_workflow = StateGraph(SupervisorState)

# Add nodes
supervisor_workflow.add_node("supervisor", supervisor_node)
supervisor_workflow.add_node("exercise", exercise_node)
supervisor_workflow.add_node("nutrition", nutrition_node)
supervisor_workflow.add_node("sleep", sleep_node)
supervisor_workflow.add_node("stress", stress_node)

# Add edges: START -> supervisor
supervisor_workflow.add_edge(START, "supervisor")

# Conditional routing from supervisor to specialists
supervisor_workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "exercise": "exercise",
        "nutrition": "nutrition",
        "sleep": "sleep",
        "stress": "stress",
    }
)

# KEY FIX: Each specialist goes directly to END (no looping!)
supervisor_workflow.add_edge("exercise", END)
supervisor_workflow.add_edge("nutrition", END)
supervisor_workflow.add_edge("sleep", END)
supervisor_workflow.add_edge("stress", END)

# Compile
supervisor_graph = supervisor_workflow.compile()

print("Supervisor multi-agent system built!")
print("Flow: User -> Supervisor -> Specialist -> END")
Supervisor multi-agent system built!
Flow: User -> Supervisor -> Specialist -> END
# Visualize the graph
try:
    from IPython.display import display, Image
    display(Image(supervisor_graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph: {e}")
    print("\nGraph structure:")
    print(supervisor_graph.get_graph().draw_ascii())

# Test the supervisor system
print("Testing Supervisor Multi-Agent System")
print("=" * 50)

response = supervisor_graph.invoke({
    "messages": [HumanMessage(content="What exercises can help with lower back pain?")]
})

print("\nFinal Response:")
print("=" * 50)
print(response["messages"][-1].content)
Testing Supervisor Multi-Agent System
==================================================
[Supervisor GPT-5.2] Routing to: exercise
  Reason: The user is asking about exercises to help with lower back pain, which is primarily a fitness/physical activity and movement topic best handled by the exercise specialist.
[EXERCISE Agent] Processing request...
[EXERCISE Agent] Response complete.

Final Response:
==================================================
[EXERCISE SPECIALIST]

Here are some effective exercises that can help alleviate lower back pain:

1. **Cat-Cow Stretch**: 
   - Start on your hands and knees.
   - Alternate between arching your back up (Cat) and letting it sag down (Cow).
   - Perform 10-15 repetitions.

2. **Bird Dog**: 
   - From a hands-and-knees position, extend your opposite arm and leg while keeping your core engaged.
   - Hold for 5 seconds, then switch sides.
   - Do 10 repetitions per side.

3. **Partial Crunches**: 
   - Lie on your back with your knees bent and arms crossed over your chest.
   - Tighten your stomach muscles and raise your shoulders off the floor.
   - Hold briefly, then lower. Aim for 8-12 repetitions.

4. **Knee-to-Chest Stretch**: 
   - Lie on your back and pull one knee toward your chest while keeping the other foot flat on the floor.
   - Hold for 15-30 seconds, then switch legs.

Incorporating these exercises into your routine can help strengthen your back and alleviate pain. Always consult with a healthcare professional before starting any new exercise program, especially if you have chronic pain.
# Test with a nutrition question
print("Testing with nutrition question")
print("=" * 50)

response = supervisor_graph.invoke({
    "messages": [HumanMessage(content="What should I eat for better gut health?")]
})

print("\nFinal Response:")
print("=" * 50)
print(response["messages"][-1].content)
Testing with nutrition question
==================================================
[Supervisor GPT-5.2] Routing to: nutrition
  Reason: The user is asking what to eat to improve gut health, which is primarily a diet and food-selection question best handled by the nutrition specialist.
[NUTRITION Agent] Processing request...
[NUTRITION Agent] Response complete.

Final Response:
==================================================
[NUTRITION SPECIALIST]

To improve your gut health, consider incorporating the following foods into your diet:

1. **Fermented Foods**: These are rich in probiotics, which support gut flora. Examples include:
   - Yogurt
   - Kefir
   - Sauerkraut
   - Kimchi
   - Miso
   - Kombucha

2. **Prebiotic Foods**: These help nourish the beneficial bacteria in your gut. Include:
   - Garlic
   - Onions
   - Bananas
   - Asparagus

3. **Fiber-Rich Foods**: Aim for 25-35 grams of fiber daily from sources like:
   - Whole grains (oats, brown rice)
   - Legumes (beans, lentils)
   - Vegetables (broccoli, carrots)

4. **Bone Broth**: This can help support gut lining health.

5. **Herbs and Spices**: Ginger and peppermint can aid digestion.

Additionally, remember to stay hydrated, eat slowly, manage stress, and limit processed foods for optimal gut health.
Task 4: Adding Tavily Search for Web Research
Sometimes the wellness knowledge base doesn't have the latest information. Let's add Tavily Search to allow agents to search the web for current information.

Documentation:

Tavily Search Tool
Tavily API Docs
# Create a Tavily search tool (using updated langchain-tavily package)
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(
    max_results=3,
    topic="general"
)

print(f"Tavily search tool created: {tavily_search.name}")
Tavily search tool created: tavily_search
# Test Tavily search
search_results = tavily_search.invoke("latest research on benefits of morning exercise 2024")
print("Tavily Search Results:")
print("-" * 50)

for result in search_results['results'][:2]:
    print(f"\nTitle: {result.get('title', 'N/A')}")
    print(f"URL: {result.get('url', 'N/A')}")
    print(f"Content: {result.get('content', 'N/A')[:200]}...")
Tavily Search Results:
--------------------------------------------------

Title: Differential benefits of 12-week morning vs. evening ...
URL: https://www.nature.com/articles/s41598-025-02659-8
Content: by B Shen · 2025 · Cited by 11 — Morning exercise (6–8 am) is particularly effective for rapid body fat reduction, lowering plasma cholesterol and triglycerides, and advancing sleep-wake cycle....

Title: This Might Be the Best Time to Exercise For Better Heart ...
URL: https://www.health.com/morning-workout-heart-lung-health-11750696
Content: New research found that morning workouts may boost cardiorespiratory fitness. Working out at the same time each day was also associated with...
# Create a research agent that can search both the knowledge base AND the web
@tool
def search_wellness_kb(query: str) -> str:
    """Search the local wellness knowledge base for established health information.
    Use this first for general wellness questions.
    """
    results = retriever.invoke(query)
    if not results:
        return "No information found in knowledge base."
    return "\n\n".join([f"[KB Source {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])

@tool
def search_web_current(query: str) -> str:
    """Search the web for current/recent health and wellness information.
    Use this when you need the latest research, news, or information not in the knowledge base.
    """
    response = tavily_search.invoke(query)
    if not response or not response.get('results'):
        return "No web results found."
    formatted = []
    for i, r in enumerate(response['results'][:3]):
        formatted.append(f"[Web Source {i+1}]: {r.get('content', 'N/A')}\nURL: {r.get('url', 'N/A')}")
    return "\n\n".join(formatted)

# Create a research agent with both tools (using create_agent)
research_agent = create_agent(
    model=specialist_llm,
    tools=[search_wellness_kb, search_web_current],
    system_prompt="""You are a Wellness Research Agent. You have access to both a curated knowledge base 
and web search. Use the knowledge base for established information and web search for 
current/recent updates. Always cite your sources."""
)

print("Research agent with web search created (using create_agent)!")
Research agent with web search created (using create_agent)!
# Test the research agent
print("Testing Research Agent (KB + Web)")
print("=" * 50)

response = research_agent.invoke({
    "messages": [HumanMessage(content="What are the benefits of cold water immersion for recovery?")]
})

print("\nResearch Agent Response:")
print(response["messages"][-1].content)
Testing Research Agent (KB + Web)
==================================================

Research Agent Response:
Cold water immersion (CWI), commonly known as ice baths or cold water therapy, has gained popularity as a recovery technique, particularly among athletes. Here are some of the key benefits associated with cold water immersion for recovery:

1. **Reduction of Muscle Soreness**: CWI has been shown to help reduce delayed onset muscle soreness (DOMS), which is the muscle pain and stiffness that often occurs after intense exercise. This can lead to quicker recovery times and allow individuals to return to their training regimens sooner.

2. **Decreased Inflammation**: Cold water immersion can help reduce inflammation and cellular damage following strenuous exercise. The cold temperature constricts blood vessels, which can help limit swelling and tissue breakdown.

3. **Improved Recovery Time**: Athletes often report feeling less fatigued and more recovered after using cold water immersion. This subjective improvement in recovery can be beneficial for maintaining performance levels during training and competition.

4. **Psychological Benefits**: The practice of cold water immersion can also provide psychological benefits, such as improved mood and reduced feelings of fatigue. The shock of cold water can stimulate the release of endorphins, which may enhance feelings of well-being.

5. **Enhanced Circulation**: After exiting the cold water, the body warms up, leading to a rush of blood flow. This can help deliver nutrients to muscles and aid in the recovery process.

6. **Potential Hormonal Benefits**: Regular exposure to cold water may influence hormone levels, including increased norepinephrine, which can enhance mood and focus.

While cold water immersion has many benefits, it's important to approach it with caution, especially for those with certain medical conditions. Gradual exposure and proper techniques are recommended to avoid shock or hyperventilation (Web Source 1, Web Source 2, Web Source 3).

For more detailed information, you can refer to the sources:
- [Chipperfield Physio](https://www.chipperfieldphysio.ca/blog-1/benefits-of-cold-water-immersion)
- [UF Health](https://ufhealthjax.org/stories/2024/the-benefits-of-cold-water-immersion-therapy)
- [ACSM](https://acsm.org/cold-water-immersion-friend-froze/)
❓ Question #1:
In the Supervisor pattern, the supervisor routes requests to specialist agents. What are the advantages and disadvantages of having agents loop back to the supervisor after responding, versus having them respond directly to the user?

Answer:
Looping specialists back to the supervisor gives you a controlled “manager” layer. The main advantage is quality control and consistency: the supervisor can normalize tone, remove contradictions, enforce policies (safety/compliance), add missing context, and decide whether another specialist should be consulted. It also helps when you want aggregation (e.g., combine nutrition + sleep advice into one coherent plan) or when you need central logging/telemetry and a single place to apply guardrails (tool limits, citations, formatting rules). The downside is you pay for it: extra latency + extra tokens every time you bounce back, and you can accidentally create loops (“supervisor re-routes” repeatedly) unless you design clear stopping rules. Another risk is “telephone game” degradation: the specialist’s answer can get paraphrased/edited and lose nuance, and the supervisor can overrule a correct specialist response with a worse rewrite.

Having specialists respond directly to the user is simpler and often best for “single-domain” questions. The advantages are speed, lower cost, and fewer moving parts. It also reduces chances of orchestration bugs and avoids the supervisor “overprocessing” a good answer. The disadvantages are that you lose the supervisor’s final pass: responses can be inconsistent across specialists (tone, structure, disclaimers), you have weaker centralized enforcement (unless you duplicate policies in each specialist), and multi-domain questions become awkward because no one is responsible for merging answers or deciding if a follow-up handoff is needed.

A practical rule: if your system’s goal is one best answer per query and you expect mostly single-topic questions, direct-to-user is usually better. If your goal is consistent, policy-controlled, multi-step or multi-domain outcomes, loop-back to supervisor is worth it—provided you enforce strict stop conditions and limit re-routing.

❓ Question #2:
We added Tavily web search alongside the knowledge base. In what scenarios would you want to restrict an agent to only use the knowledge base (no web search)? What are the trade-offs between freshness and reliability?

Answer:
You would restrict an agent to only the knowledge base when correctness, consistency, and control matter more than recency.

Typical scenarios include: i. Regulated or high-risk domains (health guidance, finance, legal, HR policies) where answers must align with vetted, approved content and you cannot afford unverified claims. ii. Internal company knowledge (runbooks, SOPs, product behavior, pricing, policies) where the web is either irrelevant or actively misleading. iii. Deterministic workflows where repeatability is important (training assistants, onboarding bots, exams, assessments). iv. Security-sensitive systems where web content could introduce prompt injection, data exfiltration, or policy bypasses. v. Teaching / grounding phases where you want the agent to reflect what you know, not what the internet says today.

In these cases, the knowledge base gives you high reliability with curated content, stable answers over time, traceable sources, predictable behavior. The cost is freshness: information can become outdated, new research, news, or changes are missed until you update the KB, the agent may confidently give answers that are no longer optimal.

Allowing Tavily web search flips the trade-off: You gain freshness (latest research, news, trends), you reduce staleness and blind spots, the agent can answer questions your KB never anticipated. But you lose some reliability: sources vary in quality, facts may conflict, content may be speculative or incorrect, higher risk of prompt injection or subtle instruction leakage, harder to guarantee consistency across runs

In practice, the best pattern is tiered access:Default to the knowledge base first for stable, trusted information. Escalate to web search only when freshness is explicitly required (e.g., “latest”, “recent”, “2024 research”) or when the KB returns low confidence

Clearly label or cite web-derived content so users understand it is less controlled

Summary:

KB-only = reliability, safety, consistency

Web search = freshness, coverage, adaptability

🏗️ Activity #1: Add a Custom Specialist Agent
Add a new specialist agent to the supervisor system. Ideas:

Habits Agent: Helps with habit formation and routines
Hydration Agent: Focuses on water intake and hydration
Lifestyle Agent: Addresses work-life balance and digital wellness
Requirements:

Create a specialized search tool for your agent's domain
Create the specialist agent with an appropriate system prompt
Add the agent to the supervisor graph
Update the routing logic
Test with relevant questions
Creating Hydration Specialist Agent. Responsibility includes Water intake, Hydration timing, Dehydration symptoms, Electrolytes, Hydration during exercise / heat / illness
Step 1: Create a Specialized Search Tool (Hydration). We reuse the same vector store but bias queries toward hydration concepts.

from langchain_core.tools import tool

@tool
def search_hydration_info(query: str) -> str:
    """Search hydration/water/electrolytes/dehydration info from the wellness knowledge base.
    Use for questions about water intake, hydration timing, electrolytes, dehydration symptoms.
    """
    results = retriever.invoke(f"hydration water fluids electrolytes dehydration {query}")
    if not results:
        return "No hydration information found in the knowledge base."
    return "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(results)])
Step 2: Create the Hydration Specialist Agent. Using GPT-4o-mini, same as other specialists.

from langchain.agents import create_agent

hydration_agent = create_agent(
    model=specialist_llm,                 # GPT-4o-mini in your notebook
    tools=[search_hydration_info],
    system_prompt=(
        "You are a Hydration Specialist. Help users with water intake, hydration timing, "
        "electrolytes, dehydration prevention, and hydration during workouts. "
        "Always search the knowledge base before answering. Be concise and actionable."
    )
)
print("Hydration agent created!")
Hydration agent created!
Step 3: Wrap the Agent as a LangGraph Node: Following the same pattern as other agents for consistency.

hydration_node = create_agent_node(hydration_agent, "hydration")
print("Hydration node created!")
Hydration node created!
Step 4: Update RouterOutput + routing_llm (Supervisor Structured Output).

from pydantic import BaseModel
from typing import Literal

class RouterOutput(BaseModel):
    """The supervisor's routing decision."""
    next: Literal["exercise", "nutrition", "sleep", "stress", "hydration"]
    reasoning: str

# Re-create structured router with updated schema
routing_llm = supervisor_llm.with_structured_output(RouterOutput)
print("RouterOutput updated to include hydration!")
RouterOutput updated to include hydration!
Step 5: Update Supervisor Prompt (so it knows hydration exists)

from langchain_core.prompts import ChatPromptTemplate

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Wellness Supervisor coordinating a team of specialist agents.

Your team:
- exercise: Handles fitness, workouts, physical activity, movement questions
- nutrition: Handles diet, meal planning, healthy eating, food questions
- hydration: Handles water intake, hydration timing, electrolytes, dehydration questions
- sleep: Handles sleep quality, insomnia, rest, recovery questions
- stress: Handles stress management, mindfulness, mental wellness, anxiety questions

Based on the user's question, decide which ONE specialist should respond.
Choose the most relevant specialist for the primary topic of the question."""),
    ("human", "User question: {question}\n\nWhich specialist should handle this?")
])

print("Supervisor prompt updated!")
Supervisor prompt updated!
Step 6:Rebuild the Supervisor Graph from scratch (with hydration included)

from langgraph.graph import StateGraph, START, END

# route_to_agent is your existing helper:
# def route_to_agent(state: SupervisorState) -> str:
#     return state["next"]

supervisor_workflow_v2 = StateGraph(SupervisorState)

# Add nodes
supervisor_workflow_v2.add_node("supervisor", supervisor_node)
supervisor_workflow_v2.add_node("exercise", exercise_node)
supervisor_workflow_v2.add_node("nutrition", nutrition_node)
supervisor_workflow_v2.add_node("sleep", sleep_node)
supervisor_workflow_v2.add_node("stress", stress_node)
supervisor_workflow_v2.add_node("hydration", hydration_node)

# START -> supervisor
supervisor_workflow_v2.add_edge(START, "supervisor")

# Supervisor -> one specialist
supervisor_workflow_v2.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "exercise": "exercise",
        "nutrition": "nutrition",
        "sleep": "sleep",
        "stress": "stress",
        "hydration": "hydration",
    }
)

# Specialists -> END
supervisor_workflow_v2.add_edge("exercise", END)
supervisor_workflow_v2.add_edge("nutrition", END)
supervisor_workflow_v2.add_edge("sleep", END)
supervisor_workflow_v2.add_edge("stress", END)
supervisor_workflow_v2.add_edge("hydration", END)

# Compile
supervisor_graph_v2 = supervisor_workflow_v2.compile()

print("Supervisor multi-agent system rebuilt with Hydration!")
print("Flow: User -> Supervisor -> Specialist -> END")
Supervisor multi-agent system rebuilt with Hydration!
Flow: User -> Supervisor -> Specialist -> END
Step 7: Test the New Agent: Using a clearly hydration-focused query.

from langchain_core.messages import HumanMessage

response = supervisor_graph_v2.invoke({
    "messages": [HumanMessage(content="How much water should I drink during a workout?")]
})

print(response["messages"][-1].content)
[Supervisor GPT-5.2] Routing to: hydration
  Reason: The question is specifically about water intake during exercise, including amounts and timing while working out—this falls squarely under hydration guidance.
[HYDRATION Agent] Processing request...
[HYDRATION Agent] Response complete.
[HYDRATION SPECIALIST]

During a workout, it's generally recommended to drink about 7-10 ounces (200-300 mL) of water every 10-20 minutes. This can vary based on factors like the intensity of the workout, climate, and individual sweat rates. 

Make sure to hydrate before, during, and after your workout to maintain optimal performance and prevent dehydration.
Step 8 : The updated graph visualization again

try:
    from IPython.display import display, Image
    display(Image(supervisor_graph_v2.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Could not display graph:", e)
    print(supervisor_graph_v2.get_graph().draw_ascii())

🤝 Breakout Room #2
Handoffs & Context Engineering
Task 5: Agent Handoffs Pattern
The Handoffs Pattern allows agents to transfer control to each other based on the conversation context. Unlike the supervisor pattern, agents decide themselves when to hand off.

    User Question
         │
         ▼
    ┌─────────┐    "I need nutrition help"   ┌─────────┐
    │ Fitness │ ─────────────────────────────► Nutrition│
    │  Agent  │                               │  Agent  │
    └─────────┘ ◄───────────────────────────── └─────────┘
                 "Back to fitness questions"
Documentation:

LangGraph Agent Handoffs
# Create handoff tools that agents can use to transfer control
# Each tool returns a special HANDOFF string that the graph will detect

@tool
def transfer_to_exercise(reason: str) -> str:
    """Transfer to Exercise Specialist for fitness, workouts, and physical activity questions.
    
    Args:
        reason: Why you're transferring to this specialist
    """
    return f"HANDOFF:exercise:{reason}"

@tool
def transfer_to_nutrition(reason: str) -> str:
    """Transfer to Nutrition Specialist for diet, meal planning, and food questions.
    
    Args:
        reason: Why you're transferring to this specialist
    """
    return f"HANDOFF:nutrition:{reason}"

@tool
def transfer_to_sleep(reason: str) -> str:
    """Transfer to Sleep Specialist for sleep quality, insomnia, and rest questions.
    
    Args:
        reason: Why you're transferring to this specialist
    """
    return f"HANDOFF:sleep:{reason}"

@tool
def transfer_to_stress(reason: str) -> str:
    """Transfer to Stress Management Specialist for stress, anxiety, and mindfulness questions.
    
    Args:
        reason: Why you're transferring to this specialist
    """
    return f"HANDOFF:stress:{reason}"

print("Handoff tools created!")
Handoff tools created!
# Create agents with handoff capabilities (using create_agent)

exercise_handoff_agent = create_agent(
    model=specialist_llm,
    tools=[
        search_exercise_info,
        transfer_to_nutrition,
        transfer_to_sleep,
        transfer_to_stress
    ],
    system_prompt="""You are an Exercise Specialist. Answer fitness and workout questions.
If the user's question is better suited for another specialist, use the appropriate transfer tool.
Always search the knowledge base before answering exercise questions."""
)

nutrition_handoff_agent = create_agent(
    model=specialist_llm,
    tools=[
        search_nutrition_info,
        transfer_to_exercise,
        transfer_to_sleep,
        transfer_to_stress
    ],
    system_prompt="""You are a Nutrition Specialist. Answer diet and meal planning questions.
If the user's question is better suited for another specialist, use the appropriate transfer tool.
Always search the knowledge base before answering nutrition questions."""
)

sleep_handoff_agent = create_agent(
    model=specialist_llm,
    tools=[
        search_sleep_info,
        transfer_to_exercise,
        transfer_to_nutrition,
        transfer_to_stress
    ],
    system_prompt="""You are a Sleep Specialist. Answer sleep and rest questions.
If the user's question is better suited for another specialist, use the appropriate transfer tool.
Always search the knowledge base before answering sleep questions."""
)

stress_handoff_agent = create_agent(
    model=specialist_llm,
    tools=[
        search_stress_info,
        transfer_to_exercise,
        transfer_to_nutrition,
        transfer_to_sleep
    ],
    system_prompt="""You are a Stress Management Specialist. Answer stress and mindfulness questions.
If the user's question is better suited for another specialist, use the appropriate transfer tool.
Always search the knowledge base before answering stress questions."""
)

print("Handoff-enabled agents created (using create_agent)!")
Handoff-enabled agents created (using create_agent)!
# Build the handoff graph with transfer limit to prevent infinite loops

class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    transfer_count: int  # Track transfers to prevent infinite loops

MAX_TRANSFERS = 2  # Maximum number of handoffs allowed

def parse_handoff(content: str) -> tuple[bool, str, str]:
    """Parse a handoff from agent response."""
    if "HANDOFF:" in content:
        parts = content.split("HANDOFF:")[1].split(":")
        return True, parts[0], parts[1] if len(parts) > 1 else ""
    return False, "", ""

def create_handoff_node(agent, name: str):
    """Create a node that can handle handoffs."""
    def node(state: HandoffState):
        print(f"[{name.upper()} Agent] Processing...")
        result = agent.invoke({"messages": state["messages"]})
        last_message = result["messages"][-1]
        
        # Check for handoff in tool messages (only if under transfer limit)
        if state["transfer_count"] < MAX_TRANSFERS:
            for msg in result["messages"]:
                if hasattr(msg, 'content') and "HANDOFF:" in str(msg.content):
                    is_handoff, target, reason = parse_handoff(str(msg.content))
                    if is_handoff:
                        print(f"[{name.upper()}] Handing off to {target}: {reason}")
                        return {
                            "messages": [AIMessage(content=f"[{name}] Transferring to {target} specialist: {reason}")],
                            "current_agent": target,
                            "transfer_count": state["transfer_count"] + 1
                        }
        
        # No handoff (or limit reached), return final response
        response = AIMessage(
            content=f"[{name.upper()} SPECIALIST]\n\n{last_message.content}",
            name=name
        )
        print(f"[{name.upper()} Agent] Response complete.")
        return {"messages": [response], "current_agent": "done", "transfer_count": state["transfer_count"]}
    
    return node

# Create nodes
exercise_handoff_node = create_handoff_node(exercise_handoff_agent, "exercise")
nutrition_handoff_node = create_handoff_node(nutrition_handoff_agent, "nutrition")
sleep_handoff_node = create_handoff_node(sleep_handoff_agent, "sleep")
stress_handoff_node = create_handoff_node(stress_handoff_agent, "stress")

print("Handoff nodes created!")
Handoff nodes created!
# Build the handoff graph with initial routing (using GPT-5.2)

def entry_router(state: HandoffState):
    """Initial routing based on the user's question (using GPT-5.2)."""
    user_question = state['messages'][-1].content
    
    router_prompt = f"""Based on this question, which specialist should handle it?
Options: exercise, nutrition, sleep, stress

Question: {user_question}

Respond with just the specialist name (one word)."""
    
    response = supervisor_llm.invoke(router_prompt)
    agent = response.content.strip().lower()
    
    # Validate
    if agent not in ["exercise", "nutrition", "sleep", "stress"]:
        agent = "stress"  # Default to stress for general wellness
    
    print(f"[Router GPT-5.2] Initial routing to: {agent}")
    return {"current_agent": agent, "transfer_count": 0}

def route_by_current_agent(state: HandoffState) -> str:
    """Route based on current_agent field."""
    return state["current_agent"]

# Build graph
handoff_workflow = StateGraph(HandoffState)

# Add nodes
handoff_workflow.add_node("router", entry_router)
handoff_workflow.add_node("exercise", exercise_handoff_node)
handoff_workflow.add_node("nutrition", nutrition_handoff_node)
handoff_workflow.add_node("sleep", sleep_handoff_node)
handoff_workflow.add_node("stress", stress_handoff_node)

# Entry point
handoff_workflow.add_edge(START, "router")

# Router to agents
handoff_workflow.add_conditional_edges(
    "router",
    route_by_current_agent,
    {"exercise": "exercise", "nutrition": "nutrition", "sleep": "sleep", "stress": "stress"}
)

# Agents can handoff to each other or end
for agent_name in ["exercise", "nutrition", "sleep", "stress"]:
    handoff_workflow.add_conditional_edges(
        agent_name,
        route_by_current_agent,
        {
            "exercise": "exercise",
            "nutrition": "nutrition", 
            "sleep": "sleep",
            "stress": "stress",
            "done": END
        }
    )

# Compile
handoff_graph = handoff_workflow.compile()

print("Handoff multi-agent system built!")
Handoff multi-agent system built!
# Visualize the handoff graph
try:
    from IPython.display import display, Image
    display(Image(handoff_graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph: {e}")
    print("\nGraph structure:")
    print(handoff_graph.get_graph().draw_ascii())

# Test the handoff system
print("Testing Handoff System")
print("=" * 50)

response = handoff_graph.invoke({
    "messages": [HumanMessage(content="I'm stressed and can't sleep. What should I do?")],
    "current_agent": "",
    "transfer_count": 0
})

print("\n" + "=" * 50)
print("FINAL RESPONSE:")
print("=" * 50)
print(response["messages"][-1].content)
Testing Handoff System
==================================================
[Router GPT-5.2] Initial routing to: sleep
[SLEEP Agent] Processing...
[SLEEP] Handing off to stress: The issue involves stress and anxiety, which is better suited for a stress management specialist.
[STRESS Agent] Processing...
[STRESS] Handing off to sleep: User is experiencing stress and difficulty sleeping.
[SLEEP Agent] Processing...
[SLEEP Agent] Response complete.

==================================================
FINAL RESPONSE:
==================================================
[SLEEP SPECIALIST]

I've transferred your concern to a stress management specialist who can help you with your stress and sleep issues. They will provide you with the appropriate guidance and support.
Task 6: Building a Wellness Agent Team
Now let's combine what we've learned to build a complete wellness team that can:

Handle complex multi-domain questions
Search both the knowledge base and the web
Maintain conversation context
Provide comprehensive wellness advice
# Create a unified wellness team with memory
from langgraph.checkpoint.memory import MemorySaver

# Add memory to the supervisor graph
memory = MemorySaver()

supervisor_with_memory = supervisor_workflow.compile(checkpointer=memory)

print("Supervisor with memory created!")
Supervisor with memory created!
# Test multi-turn conversation
thread_id = "wellness-session-1"
config = {"configurable": {"thread_id": thread_id}}

print("Multi-turn Conversation Test")
print("=" * 50)

# First question
response1 = supervisor_with_memory.invoke(
    {"messages": [HumanMessage(content="What's a good morning routine for energy?")]},
    config=config
)
print("\n[Turn 1 Response]:")
print(response1["messages"][-1].content[:500])
Multi-turn Conversation Test
==================================================
[Supervisor GPT-5.2] Routing to: sleep
  Reason: A morning routine for energy is most strongly influenced by sleep quality, wake timing, and early-day light exposure; the sleep specialist can anchor the routine around circadian cues and recovery, with optional nods to movement, hydration, and nutrition.
[SLEEP Agent] Processing request...
[SLEEP Agent] Response complete.

[Turn 1 Response]:
[SLEEP SPECIALIST]

A good morning routine for energy includes the following steps:

1. **Wake at a Consistent Time**: Try to get up at the same time every day to regulate your body's internal clock.
2. **Hydrate**: Drink a glass of water immediately after waking to rehydrate your body.
3. **Light Movement**: Engage in 5-10 minutes of stretching or light exercise to get your blood flowing.
4. **Healthy Breakfast**: Eat a nutritious breakfast that includes protein, healthy fats, and whole grains.
# Follow-up question (should remember context)
response2 = supervisor_with_memory.invoke(
    {"messages": [HumanMessage(content="What should I eat as part of that routine?")]},
    config=config
)
print("\n[Turn 2 Response]:")
print(response2["messages"][-1].content[:500])
[Supervisor GPT-5.2] Routing to: nutrition
  Reason: The user is asking what to eat as part of a routine, which is a diet/meal planning question best handled by the nutrition specialist.
[NUTRITION Agent] Processing request...
[NUTRITION Agent] Response complete.

[Turn 2 Response]:
[NUTRITION SPECIALIST]

For a healthy breakfast that boosts energy, consider these options:

1. **Overnight Oats**: Combine rolled oats with milk or yogurt, and top with berries and nuts for added fiber and healthy fats.
2. **Smoothie Bowl**: Blend spinach, banana, and a scoop of protein powder, then top with granola and seeds.
3. **Avocado Toast**: Whole grain bread topped with smashed avocado, a sprinkle of salt, and a poached egg for protein.
4. **Greek Yogurt Parfait**: Layer Greek yogurt wi
Task 7: Context Engineering & Optimization
As conversations grow, we need to manage context carefully. Key principles:

Context Window as Prime Real Estate: Only include what's necessary
Summarization: Compress long conversations
Selective Retrieval: Don't retrieve everything, just what's relevant
Context Rot: More tokens doesn't mean better performance
# Implement a context summarization function (using GPT-4o-mini for cost efficiency)

def summarize_conversation(messages: list[BaseMessage], max_messages: int = 6) -> list[BaseMessage]:
    """Summarize older messages to manage context length."""
    if len(messages) <= max_messages:
        return messages
    
    # Keep the first message (original question) and last few messages
    old_messages = messages[1:-max_messages+1]
    recent_messages = messages[-max_messages+1:]
    
    # Summarize old messages
    summary_prompt = f"""Summarize this conversation history in 2-3 sentences, 
capturing the key topics discussed and any important decisions made:

{chr(10).join([f'{m.type}: {m.content[:200]}' for m in old_messages])}"""
    
    summary = specialist_llm.invoke(summary_prompt)
    
    # Return: first message + summary + recent messages
    return [
        messages[0],
        SystemMessage(content=f"[Previous conversation summary: {summary.content}]"),
        *recent_messages
    ]

print("Context summarization function created!")
Context summarization function created!
# Demonstrate context optimization
sample_messages = [
    HumanMessage(content="I want to get healthier"),
    AIMessage(content="Great! Let's start with your goals."),
    HumanMessage(content="I want to lose weight and sleep better"),
    AIMessage(content="Here are some exercise tips..."),
    HumanMessage(content="What about diet?"),
    AIMessage(content="For nutrition, consider..."),
    HumanMessage(content="And sleep?"),
    AIMessage(content="For better sleep..."),
    HumanMessage(content="How do I manage stress?"),
]

print(f"Original messages: {len(sample_messages)}")

optimized = summarize_conversation(sample_messages, max_messages=4)
print(f"Optimized messages: {len(optimized)}")
print("\nOptimized conversation:")
for msg in optimized:
    print(f"  [{msg.type}]: {msg.content[:100]}...")
Original messages: 9
Optimized messages: 5

Optimized conversation:
  [human]: I want to get healthier...
  [system]: [Previous conversation summary: The conversation focused on the human's goals of losing weight and i...
  [human]: And sleep?...
  [ai]: For better sleep......
  [human]: How do I manage stress?...
❓ Question #3:
Compare the Supervisor pattern and the Handoffs pattern we implemented. What are the key differences in how routing decisions are made? When would you choose one pattern over the other?

Answer:
In the Supervisor pattern, routing is a centralized decision: one supervisor looks at the user question and picks one specialist to handle it. The specialists don’t decide where the conversation goes; they just answer. So the “control plane” is explicit and top-down: Supervisor → Specialist → END. This makes routing predictable and easy to govern, because there is a single place to enforce rules like “only one specialist per turn,” consistent tone/format, logging, safety constraints, and tool policies. It also makes debugging simpler because every request has a single recorded routing decision and reasoning.

In the Handoffs pattern, routing is decentralized and dynamic: after an initial entry router selects a starting agent, each specialist can decide, mid-conversation, “this belongs to someone else” and explicitly trigger a transfer using a handoff tool . So the “control plane” is distributed: Agent decides → graph detects HANDOFF → routes to next agent. This is better for multi-topic or evolving conversations because the system can move across domains as new information emerges. But it’s harder to control and debug: you can get ping-pong behavior, and behavior is less predictable because decisions are made by multiple agents with partial context and different prompts.

When to choose which: we should pick Supervisor when we want tight governance, predictability, and a clean one-shot answer per user query, or when compliance/consistency matters and the task is usually single-domain. We should pick Handoffs when you expect multi-domain, conversational problems where the “right” specialist can change as the user reveals more, and we are willing to manage the extra complexity (transfer limits, clear handoff rules, and good observability) to get smoother expertise routing.

❓ Question #4:
We discussed "Context Rot" - the idea that longer context doesn't always mean better performance. How does this principle apply to multi-agent systems? What strategies can you use to manage context effectively across multiple agents?

Answer:
“Context rot” applies even more strongly in multi-agent systems because context is not just growing within one model call, it is being copied, transformed, and re-interpreted across multiple agents. Every extra message, tool output, or handoff increases the chance that an agent attends to irrelevant or stale information, misroutes a task, or produces a lower-quality answer. So while multi-agent systems feel like they should benefit from “more context everywhere,” in practice they degrade faster if context is not tightly controlled.

In a Supervisor pattern, context rot shows up when the supervisor is given too much conversation history or too many retrieved documents. The router may overweight older turns, misclassify the user’s current intent, or choose the wrong specialist. In Handoffs, the problem is amplified: agents may inherit long, noisy histories from previous agents and make poor handoff decisions, leading to loops, redundant transfers, or diluted answers.

To manage context effectively, the first strategy is role-specific context scoping. Each agent should see only the information it needs to do its job. For example, the supervisor should primarily see the latest user question (and perhaps a short summary), not full tool outputs. Specialists should see relevant retrieved chunks, not the entire conversation or other specialists’ reasoning.

Second is summarization and compression. As conversations grow, older turns should be summarized into a short, structured system message (as you implemented) that preserves goals and decisions but removes conversational noise. This keeps intent without burning tokens.

Third is selective retrieval, not blanket retrieval. Retrieval should be filtered by domain and intent (exercise vs hydration vs sleep) so each agent only pulls a small, high-signal subset of documents. This avoids flooding agents with loosely related chunks that increase cognitive load.

Fourth is bounded interaction. In Handoffs, hard limits like MAX_TRANSFERS, clear “done” states, and explicit handoff reasons prevent runaway context growth and agent ping-pong.

Finally, we should treat context as infrastructure, not a side effect. We should design explicit policies for what gets passed between agents, what gets summarized, what gets dropped, and when memory is consulted. In multi-agent systems, reliability comes less from bigger context windows and more from disciplined context construction and pruning.

🏗️ Activity #2: Implement Hierarchical Teams
Build a Hierarchical Agent System where a top-level supervisor manages multiple team supervisors, each with their own specialist agents.

Requirements:
Create a Wellness Director (top-level supervisor using GPT-5.2) that:

Receives user questions and determines which team should handle it
Routes to either the "Physical Wellness Team" or "Mental Wellness Team"
Aggregates final responses from teams
Create two Team Supervisors:

Physical Wellness Team Lead: Manages Exercise Agent and Nutrition Agent
Mental Wellness Team Lead: Manages Sleep Agent and Stress Agent
Implement the hierarchical routing:

User question → Wellness Director → Team Lead → Specialist Agent → Response
Test with questions that require different teams:

"What exercises help with weight loss?" (Physical team)
"How can I improve my sleep when stressed?" (Mental team)
Architecture:
                    ┌─────────────────────┐
                    │  Wellness Director  │
                    │     (GPT-5.2)       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │  Physical Wellness  │          │  Mental Wellness    │
   │    Team Lead        │          │    Team Lead        │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                 │
       ┌──────┴──────┐                   ┌──────┴──────┐
       │             │                   │             │
       ▼             ▼                   ▼             ▼
  ┌─────────┐  ┌──────────┐        ┌─────────┐  ┌─────────┐
  │Exercise │  │Nutrition │        │  Sleep  │  │ Stress  │
  │  Agent  │  │  Agent   │        │  Agent  │  │  Agent  │
  └─────────┘  └──────────┘        └─────────┘  └─────────┘
Documentation:

LangGraph Hierarchical Teams
### YOUR CODE HERE ###

# Step 1: Create Team Supervisors (using GPT-5.2 for routing)
# These manage routing within their teams

class TeamRouterOutput(BaseModel):
    """Team supervisor routing decision."""
    next: str  # The specialist to route to within the team
    reasoning: str

# Physical Wellness Team Lead
physical_team_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Physical Wellness Team Lead.
Your team has two specialists:
- exercise: Handles fitness, workouts, and physical activity
- nutrition: Handles diet, meal planning, and healthy eating

Route to the most appropriate specialist for the user's question."""),
    ("human", "Question: {question}")
])

# Mental Wellness Team Lead  
mental_team_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Mental Wellness Team Lead.
Your team has two specialists:
- sleep: Handles sleep quality, insomnia, and rest
- stress: Handles stress management, mindfulness, and mental wellness

Route to the most appropriate specialist for the user's question."""),
    ("human", "Question: {question}")
])

# Step 2: Create the Wellness Director (top-level, using GPT-5.2)
class DirectorRouterOutput(BaseModel):
    """Director routing decision to teams."""
    team: Literal["physical", "mental"]
    reasoning: str

director_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Wellness Director overseeing two teams:
- physical: Physical Wellness Team (exercise, nutrition)
- mental: Mental Wellness Team (sleep, stress)

Route to the appropriate team based on the user's question."""),
    ("human", "Question: {question}")
])

# Step 3: Build the hierarchical graph
# Hint: You'll need nested graphs or a state that tracks the current level

# Step 4: Test the hierarchical system
# test_question = "What exercises help with weight loss?"
# response = hierarchical_graph.invoke({"messages": [HumanMessage(content=test_question)]})
# print(response["messages"][-1].content)

# =========================
# Step 3: Build the hierarchical graph
# =========================

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# A) Create structured routers from your prompts
director_router = supervisor_llm.with_structured_output(DirectorRouterOutput)
team_router_llm = supervisor_llm.with_structured_output(TeamRouterOutput)

# B) Define state that carries routing decisions
class HierState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    team: str   # "physical" or "mental"
    next: str   # "exercise"/"nutrition" or "sleep"/"stress"

def last_user_question(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""

# C) Director node: decides team
def director_node(state: HierState):
    q = last_user_question(state["messages"])
    out = director_router.invoke(director_prompt.invoke({"question": q}))
    return {"team": out.team}

# D) Physical team lead: decides exercise vs nutrition
def physical_lead_node(state: HierState):
    q = last_user_question(state["messages"])
    out = team_router_llm.invoke(physical_team_prompt.invoke({"question": q}))
    nxt = out.next.strip().lower()
    if nxt not in {"exercise", "nutrition"}:
        nxt = "exercise"
    return {"next": nxt}

# E) Mental team lead: decides sleep vs stress
def mental_lead_node(state: HierState):
    q = last_user_question(state["messages"])
    out = team_router_llm.invoke(mental_team_prompt.invoke({"question": q}))
    nxt = out.next.strip().lower()
    if nxt not in {"sleep", "stress"}:
        nxt = "stress"
    return {"next": nxt}

# F) Routing helpers for conditional edges
def route_by_team(state: HierState) -> str:
    return state["team"]

def route_by_next(state: HierState) -> str:
    return state["next"]

# G) Build graph (reuse your existing specialist nodes from earlier)
hier_workflow = StateGraph(HierState)

hier_workflow.add_node("director", director_node)
hier_workflow.add_node("physical_lead", physical_lead_node)
hier_workflow.add_node("mental_lead", mental_lead_node)

# These must already exist from earlier in your notebook:
# exercise_node, nutrition_node, sleep_node, stress_node
hier_workflow.add_node("exercise", exercise_node)
hier_workflow.add_node("nutrition", nutrition_node)
hier_workflow.add_node("sleep", sleep_node)
hier_workflow.add_node("stress", stress_node)

# START -> Director
hier_workflow.add_edge(START, "director")

# Director -> Team Lead
hier_workflow.add_conditional_edges(
    "director",
    route_by_team,
    {"physical": "physical_lead", "mental": "mental_lead"}
)

# Physical lead -> Specialist
hier_workflow.add_conditional_edges(
    "physical_lead",
    route_by_next,
    {"exercise": "exercise", "nutrition": "nutrition"}
)

# Mental lead -> Specialist
hier_workflow.add_conditional_edges(
    "mental_lead",
    route_by_next,
    {"sleep": "sleep", "stress": "stress"}
)

# Specialists -> END
hier_workflow.add_edge("exercise", END)
hier_workflow.add_edge("nutrition", END)
hier_workflow.add_edge("sleep", END)
hier_workflow.add_edge("stress", END)

hierarchical_graph = hier_workflow.compile()

print("✅ hierarchical_graph compiled")

# =========================
# Step 4: Test the hierarchical system
# =========================

test_question = "What exercises help with weight loss?"
response = hierarchical_graph.invoke({"messages": [HumanMessage(content=test_question)]})
print(response["messages"][-1].content)

test_question = "How can I improve my sleep when stressed?"
response = hierarchical_graph.invoke({"messages": [HumanMessage(content=test_question)]})
print(response["messages"][-1].content)
✅ hierarchical_graph compiled
[EXERCISE Agent] Processing request...
[EXERCISE Agent] Response complete.
[EXERCISE SPECIALIST]

To aid in weight loss, a combination of different types of exercises is effective. Here are some key types:

1. **Aerobic (Cardio) Exercises**: Activities like walking, running, cycling, and swimming help burn calories and improve cardiovascular health. Aim for at least 150 minutes of moderate-intensity aerobic activity per week.

2. **Strength Training**: Incorporating bodyweight exercises (like squats, push-ups, and planks) or using weights helps build muscle, which can increase your resting metabolic rate. Aim for muscle-strengthening activities on 2 or more days per week.

3. **Flexibility and Balance Exercises**: While these may not directly contribute to weight loss, they enhance overall fitness and can prevent injuries, allowing you to maintain a consistent workout routine.

4. **Recreational Activities**: Engaging in fun activities like hiking, dancing, or sports can also contribute to calorie burning while keeping you motivated.

A balanced routine that includes these elements will be most effective for weight loss.
[SLEEP Agent] Processing request...
[SLEEP Agent] Response complete.
[SLEEP SPECIALIST]

To improve your sleep when stressed, consider the following strategies:

1. **Establish a Sleep Routine**: Go to bed and wake up at the same time every day, even on weekends.

2. **Create a Relaxing Bedtime Ritual**: Engage in calming activities before bed, such as reading, gentle stretching, or taking a warm bath.

3. **Optimize Your Sleep Environment**: Keep your bedroom cool, dark, and quiet to promote better sleep.

4. **Limit Screen Time**: Avoid screens (phones, computers, TVs) 1-2 hours before bedtime to reduce blue light exposure.

5. **Watch Your Diet**: Avoid caffeine after 2 PM, and limit alcohol and heavy meals close to bedtime.

6. **Practice Relaxation Techniques**: Try progressive muscle relaxation, meditation, or deep breathing exercises to reduce stress.

7. **Consider Natural Remedies**: Herbal teas like chamomile or valerian root may help, and magnesium supplements can be beneficial (consult a healthcare provider first).

8. **Exercise Regularly**: Engage in physical activity, but avoid vigorous exercise close to bedtime.

Implementing these practices can help enhance your sleep quality and manage stress more effectively.
Advanced Build : Personal Wellness Planner with File I/O
Step 1 : Create the plans/ folder + File System Tools

import os
from langchain_core.tools import tool

PLANS_DIR = "plans"
os.makedirs(PLANS_DIR, exist_ok=True)

def _safe_path(filename: str) -> str:
    """
    Restrict file operations to ./plans only.
    Only allow simple filenames like 'wellness_plan_weight_loss.md'
    """
    filename = filename.strip().replace("\\", "/")
    if "/" in filename or filename.startswith("."):
        raise ValueError("Filename must be a simple file name (no folders), e.g. wellness_plan.md")
    return os.path.join(PLANS_DIR, filename)

@tool
def save_wellness_plan(filename: str, content: str) -> str:
    """Save a wellness plan (markdown) to plans/<filename>."""
    path = _safe_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved plan to {path}"

@tool
def load_wellness_plan(filename: str) -> str:
    """Load a wellness plan (markdown) from plans/<filename>."""
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"Plan not found: {filename}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@tool
def list_saved_plans() -> str:
    """List all saved plans in the plans/ folder."""
    files = sorted([f for f in os.listdir(PLANS_DIR) if os.path.isfile(os.path.join(PLANS_DIR, f))])
    if not files:
        return "No plans saved yet in plans/."
    return "Saved plans:\n" + "\n".join(f"- {f}" for f in files)

@tool
def append_to_plan(filename: str, section: str, content: str) -> str:
    """
    Append content to an existing plan under a section header.
    If section header doesn't exist, it will be added.
    """
    path = _safe_path(filename)

    existing = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read()

    section_header = f"## {section}".strip()
    addition = f"\n\n{section_header}\n{content.strip()}\n"

    if section_header in existing:
        updated = existing + f"\n{content.strip()}\n"
    else:
        updated = existing + addition

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    return f"Appended to {path} under section '{section_header}'"
Step 2: Create Specialist Agents (Exercise/Nutrition/Sleep/Stress)

from langchain.agents import create_agent

exercise_specialist = create_agent(
    model=specialist_llm,  # GPT-4o-mini
    tools=[search_exercise_info],
    system_prompt=(
        "You are an Exercise Specialist.\n"
        "Use the knowledge base tool before answering.\n"
        "Return a concise markdown section titled '## Exercise Plan' with a 7-day plan."
    )
)

nutrition_specialist = create_agent(
    model=specialist_llm,
    tools=[search_nutrition_info],
    system_prompt=(
        "You are a Nutrition Specialist.\n"
        "Use the knowledge base tool before answering.\n"
        "Return a concise markdown section titled '## Nutrition Plan' with practical guidance."
    )
)

sleep_specialist = create_agent(
    model=specialist_llm,
    tools=[search_sleep_info],
    system_prompt=(
        "You are a Sleep Specialist.\n"
        "Use the knowledge base tool before answering.\n"
        "Return a concise markdown section titled '## Sleep Plan' with a 7-day plan."
    )
)

stress_specialist = create_agent(
    model=specialist_llm,
    tools=[search_stress_info],
    system_prompt=(
        "You are a Stress Management Specialist.\n"
        "Use the knowledge base tool before answering.\n"
        "Return a concise markdown section titled '## Stress Management Plan' with daily practices."
    )
)

print("✅ Specialist agents created")
✅ Specialist agents created
Step 3: Create the File Manager Agent (only agent with file tools)

file_manager_agent = create_agent(
    model=specialist_llm,  # cheap is fine
    tools=[save_wellness_plan, load_wellness_plan, list_saved_plans, append_to_plan],
    system_prompt=(
        "You are a File Manager Agent.\n"
        "You ONLY manage saving, loading, listing, and appending wellness plans in the plans/ folder.\n"
        "When asked to save, call save_wellness_plan. When asked to append, call append_to_plan.\n"
        "Do not invent file contents—use what is provided by other agents.\n"
    )
)

print("✅ File Manager agent created")
✅ File Manager agent created
Step 4 : Planner Supervisor (GPT-5.2) orchestrates the workflow

from typing import Annotated, TypedDict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class PlannerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    filename: str
    goals_md: str
    exercise_md: str
    nutrition_md: str
    sleep_md: str
    stress_md: str
    final_plan_md: str

def _user_request(state: PlannerState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""

def supervisor_make_goals(state: PlannerState):
    """
    Supervisor (GPT-5.2) extracts goals + constraints into Goals section.
    """
    user_text = _user_request(state)
    prompt = f"""
Create ONLY the markdown for these sections:

## Goals
## Constraints & Preferences

Based on this user request:
{user_text}
"""
    goals_md = supervisor_llm.invoke(prompt).content
    return {"goals_md": goals_md}

def run_exercise(state: PlannerState):
    user_text = _user_request(state)
    result = exercise_specialist.invoke({"messages": [HumanMessage(content=user_text)]})
    return {"exercise_md": result["messages"][-1].content}

def run_nutrition(state: PlannerState):
    user_text = _user_request(state)
    result = nutrition_specialist.invoke({"messages": [HumanMessage(content=user_text)]})
    return {"nutrition_md": result["messages"][-1].content}

def run_sleep(state: PlannerState):
    user_text = _user_request(state)
    result = sleep_specialist.invoke({"messages": [HumanMessage(content=user_text)]})
    return {"sleep_md": result["messages"][-1].content}

def run_stress(state: PlannerState):
    user_text = _user_request(state)
    result = stress_specialist.invoke({"messages": [HumanMessage(content=user_text)]})
    return {"stress_md": result["messages"][-1].content}

def supervisor_merge_plan(state: PlannerState):
    """
    Supervisor (GPT-5.2) merges everything into the final plan format
    similar to the example in the assignment.
    """
    generated = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
Combine the sections into ONE markdown plan following this exact structure:

# Personal Wellness Plan
Generated: {generated}

{state["goals_md"]}

{state["exercise_md"]}

{state["nutrition_md"]}

{state["stress_md"]}

{state["sleep_md"]}

## Weekly Check-in Template
- [ ] Completed workouts: _/5
- [ ] Followed nutrition plan: _/7 days
- [ ] Sleep routine followed: _/7 nights
- [ ] Stress practice done: _/7 days

## Disclaimer
This plan is informational and not medical advice. If you have medical conditions or pain, consult a clinician.
"""
    final_md = supervisor_llm.invoke(prompt).content

    # default filename if not provided
    filename = state.get("filename")
    if not filename:
        filename = f"wellness_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    return {"final_plan_md": final_md, "filename": filename}

def file_manager_save(state: PlannerState):
    """
    Call the File Manager agent to save.
    """
    filename = state["filename"]
    content = state["final_plan_md"]

    save_request = f"Save this plan as '{filename}'. Content:\n\n{content}"
    result = file_manager_agent.invoke({"messages": [HumanMessage(content=save_request)]})

    return {"messages": [AIMessage(content=result["messages"][-1].content)]}

# ---- Build the planner graph ----
planner_workflow = StateGraph(PlannerState)
planner_workflow.add_node("make_goals", supervisor_make_goals)
planner_workflow.add_node("exercise", run_exercise)
planner_workflow.add_node("nutrition", run_nutrition)
planner_workflow.add_node("stress", run_stress)
planner_workflow.add_node("sleep", run_sleep)
planner_workflow.add_node("merge", supervisor_merge_plan)
planner_workflow.add_node("save", file_manager_save)

planner_workflow.add_edge(START, "make_goals")
planner_workflow.add_edge("make_goals", "exercise")
planner_workflow.add_edge("exercise", "nutrition")
planner_workflow.add_edge("nutrition", "stress")
planner_workflow.add_edge("stress", "sleep")
planner_workflow.add_edge("sleep", "merge")
planner_workflow.add_edge("merge", "save")
planner_workflow.add_edge("save", END)

planner_graph = planner_workflow.compile()

print("✅ Planner graph compiled (Supervisor -> Specialists -> File Manager -> plans/)")
✅ Planner graph compiled (Supervisor -> Specialists -> File Manager -> plans/)
Step 5 : Test: create and save a plan (this proves File I/O works)

from langchain_core.messages import HumanMessage

request = """Create a wellness plan for someone who wants to lose weight and reduce stress.
Constraints: 30 minutes/day, vegetarian, beginner, mild lower-back discomfort.
Preferences: simple meals, home workouts, bedtime 10:30pm.
"""

out = planner_graph.invoke({
    "messages": [HumanMessage(content=request)],
    "filename": "wellness_plan_weight_loss_stress.md"
})

print(out["messages"][-1].content)
print("\n---\nSaved files:")
print(list_saved_plans.invoke({}))
The wellness plan has been successfully saved as 'wellness_plan_weight_loss_stress.md'. If you need any further assistance, feel free to ask!

---
Saved files:
Saved plans:
- plan_demo_1.md
- plan_example_1.md
- plan_example_2.md
- wellness_plan_weight_loss_stress.md
Advanced Build : Create 2 example plans in plans/
Generate plan #1 (weight loss + stress)
from langchain_core.messages import HumanMessage

req1 = """Create a wellness plan for someone who wants to lose weight and reduce stress.
Constraints: 30 minutes/day, vegetarian, beginner, mild lower-back discomfort.
Preferences: simple meals, home workouts, bedtime 10:30pm.
"""

planner_graph.invoke({
    "messages": [HumanMessage(content=req1)],
    "filename": "wellness_plan_weight_loss_stress.md"
})
{'messages': [HumanMessage(content='Create a wellness plan for someone who wants to lose weight and reduce stress.\nConstraints: 30 minutes/day, vegetarian, beginner, mild lower-back discomfort.\nPreferences: simple meals, home workouts, bedtime 10:30pm.\n', additional_kwargs={}, response_metadata={}, id='facac961-3ad1-4894-a4f7-b96a4ad98a65'),
  AIMessage(content="The wellness plan has been successfully saved as 'wellness_plan_weight_loss_stress.md'.", additional_kwargs={}, response_metadata={}, id='8fd16d43-a680-4399-b896-d92c80cf5516', tool_calls=[], invalid_tool_calls=[])],
 'filename': 'wellness_plan_weight_loss_stress.md',
 'goals_md': '## Goals\n- Lose weight in a sustainable, beginner-friendly way.\n- Reduce daily stress and improve overall mood and resilience.\n- Improve energy levels and establish consistent healthy routines.\n- Support back comfort and core stability to reduce mild lower-back discomfort.\n\n## Constraints & Preferences\n- **Time:** Maximum **30 minutes per day** for wellness activities (exercise, stress reduction, prep).\n- **Diet:** **Vegetarian**.\n- **Experience level:** **Beginner** (workouts and habit changes should start easy and progress gradually).\n- **Physical considerations:** **Mild lower-back discomfort** (low-impact, back-friendly movements; prioritize form and core/glute strength).\n- **Meal preference:** **Simple meals** (minimal ingredients, low prep complexity).\n- **Workout preference:** **Home workouts** (no gym required; minimal equipment).\n- **Sleep preference:** **Bedtime at 10:30pm** (plan should support a consistent wind-down routine).',
 'exercise_md': '## Exercise Plan\n\n### Weekly Schedule\n**Day 1: Monday**\n- **Activity**: 20-minute walk\n- **Stretching**: 10 minutes (focus on lower back stretches)\n\n**Day 2: Tuesday**\n- **Activity**: 15 minutes bodyweight exercises\n  - Squats\n  - Push-ups (knee push-ups if needed)\n  - Planks (hold for 10-15 seconds)\n\n**Day 3: Wednesday**\n- **Activity**: Rest or gentle yoga (focus on flexibility and relaxation)\n\n**Day 4: Thursday**\n- **Activity**: 20-minute walk\n- **Stretching**: 10 minutes (focus on lower back stretches)\n\n**Day 5: Friday**\n- **Activity**: 15 minutes bodyweight exercises\n  - Lunges\n  - Wall sits\n  - Glute bridges\n\n**Day 6: Saturday**\n- **Activity**: 30-minute recreational activity (choose something enjoyable like swimming, cycling, or hiking)\n\n**Day 7: Sunday**\n- **Activity**: Rest\n\n### Meal Plan\n**Breakfast Options:**\n- Overnight oats with fruits and nuts\n- Smoothie with spinach, banana, and almond milk\n\n**Lunch Options:**\n- Quinoa salad with mixed vegetables and chickpeas\n- Vegetable stir-fry with tofu and brown rice\n\n**Dinner Options:**\n- Lentil soup with whole-grain bread\n- Stuffed bell peppers with brown rice and black beans\n\n**Snacks:**\n- Fresh fruits (apples, bananas, berries)\n- Carrot sticks with hummus\n\n### Tips\n- Prepare meals in advance to save time and reduce stress.\n- Focus on hydration; drink plenty of water throughout the day.\n- Aim to finish eating at least 2-3 hours before bedtime (10:30 PM).',
 'nutrition_md': '## Nutrition Plan\n\n### Meal Suggestions\n- **Breakfast**: Overnight oats with berries and nuts.\n- **Mid-Morning Snack**: Apple with almond butter.\n- **Lunch**: Quinoa salad with mixed greens, cherry tomatoes, cucumber, and a light olive oil dressing.\n- **Afternoon Snack**: Carrot sticks with hummus.\n- **Dinner**: Stir-fried tofu with mixed vegetables (broccoli, bell peppers, carrots) served over brown rice.\n- **Evening**: Herbal tea.\n\n### Meal Prep Tips\n1. **Plan Ahead**: Choose 3-4 main dishes for the week and prepare them in batches.\n2. **Healthy Snacks**: Keep fruits, nuts, and yogurt on hand to avoid impulse eating.\n3. **Hydration**: Drink plenty of water throughout the day.\n\n## Exercise Plan\n\n### Weekly Schedule\n- **Monday**: 20-minute walk + 10 minutes of gentle stretching (Cat-Cow Stretch, Bird Dog).\n- **Tuesday**: 15 minutes of bodyweight exercises (squats, modified push-ups, planks).\n- **Wednesday**: Rest or gentle yoga.\n- **Thursday**: 20-minute walk + 10 minutes of stretching.\n- **Friday**: 15 minutes of bodyweight exercises.\n- **Saturday**: 30-minute recreational activity (swimming, cycling, or hiking).\n- **Sunday**: Rest.\n\n### Gentle Exercises for Lower Back Discomfort\n- **Cat-Cow Stretch**: 10-15 repetitions.\n- **Bird Dog**: 10 repetitions per side, holding for 5 seconds.\n\n## Stress Reduction Techniques\n- **Deep Breathing**: Inhale for 4 counts, hold for 4, exhale for 4.\n- **Progressive Muscle Relaxation**: Tense and release muscle groups from toes to head.\n- **Grounding Technique**: Identify 5 things you see, 4 you hear, 3 you feel, 2 you smell, and 1 you taste.\n- **Nature Walks**: Spend time outdoors to reduce stress.\n- **Mindfulness Practice**: Engage in mindfulness or meditation for 5-10 minutes daily.\n\n### Sleep Hygiene\n- Aim to wind down by 10:00 PM to ensure a restful sleep by 10:30 PM. Consider a calming bedtime routine, such as reading or listening to soothing music.',
 'sleep_md': '## Sleep Plan\n\n### 7-Day Wellness Plan for Weight Loss and Stress Reduction\n\n**Daily Schedule:**\n- **Morning Routine:** Start your day with a glass of water and a light breakfast.\n- **Evening Routine:** Wind down with a relaxing activity (reading, gentle stretching) before bed at 10:30 PM.\n\n### Weekly Breakdown\n\n| Day       | Activity                                                                                     | Meal Ideas                          |\n|-----------|----------------------------------------------------------------------------------------------|-------------------------------------|\n| **Monday**    | 20-minute walk + 10 minutes of Cat-Cow Stretch and Bird Dog exercises                     | Vegetable stir-fry with tofu       |\n| **Tuesday**   | 15 minutes of bodyweight exercises (squats, push-ups, planks)                             | Quinoa salad with chickpeas        |\n| **Wednesday** | Rest day or gentle yoga (focus on stretching)                                             | Lentil soup with whole grain bread |\n| **Thursday**  | 20-minute walk + 10 minutes of Cat-Cow Stretch and Bird Dog exercises                     | Stuffed bell peppers with rice     |\n| **Friday**    | 15 minutes of bodyweight exercises                                                         | Spinach and feta omelette          |\n| **Saturday**  | 30-minute recreational activity (swimming, cycling, or hiking)                            | Vegetable curry with brown rice    |\n| **Sunday**    | Rest day or light stretching                                                                | Smoothie bowl with fruits and nuts |\n\n### Additional Tips\n- **Sleep Hygiene:** Maintain a consistent sleep schedule, limit screen time before bed, and create a calming bedtime routine.\n- **Hydration:** Drink plenty of water throughout the day and include water-rich foods in your meals.\n- **Stress Management:** Incorporate relaxation techniques such as deep breathing or meditation into your daily routine.\n\nThis plan is designed to be simple and manageable, focusing on gentle exercises and easy vegetarian meals while promoting better sleep and stress reduction.',
 'stress_md': '## Stress Management Plan\n\n### Daily Practices (30 minutes/day)\n\n#### Morning Routine (10 minutes)\n- **Mindfulness Meditation**: Spend 5 minutes practicing mindfulness or deep breathing to start your day with a calm mind.\n- **Gentle Stretching**: Perform gentle stretches focusing on the lower back to alleviate discomfort. Include:\n  - **Cat-Cow Stretch**: 10-15 repetitions.\n  - **Bird Dog**: 10 repetitions per side.\n\n#### Midday Activity (10 minutes)\n- **Home Workout**: Engage in a simple bodyweight workout that includes:\n  - **Squats**: 10-15 repetitions.\n  - **Push-ups (knee or wall)**: 5-10 repetitions.\n  - **Planks**: Hold for 10-20 seconds.\n\n#### Evening Wind Down (10 minutes)\n- **Light Walk**: Take a 10-minute walk in the evening to relax and reflect on your day.\n- **Evening Stretching**: Focus on lower back stretches to relieve tension before bed.\n\n### Meal Suggestions\n- **Breakfast**: Overnight oats with fruits and nuts.\n- **Lunch**: Quinoa salad with mixed vegetables and a light vinaigrette.\n- **Dinner**: Stir-fried tofu with broccoli and brown rice.\n- **Snacks**: Fresh fruits, nuts, or yogurt.\n\n### Additional Tips\n- **Sleep Hygiene**: Aim to wind down by 10:00 PM to prepare for bed by 10:30 PM.\n- **Social Connections**: Engage with friends or family regularly to enhance emotional support.\n- **Limit Screen Time**: Reduce exposure to news and social media, especially before bedtime.\n\nThis plan is designed to help you manage stress while supporting your weight loss goals in a gentle and sustainable manner.',
 'final_plan_md': '# Personal Wellness Plan\nGenerated: 2026-02-07\n\n## Goals\n- Lose weight in a sustainable, beginner-friendly way.\n- Reduce daily stress and improve overall mood and resilience.\n- Improve energy levels and establish consistent healthy routines.\n- Support back comfort and core stability to reduce mild lower-back discomfort.\n\n## Constraints & Preferences\n- **Time:** Maximum **30 minutes per day** for wellness activities (exercise, stress reduction, prep).\n- **Diet:** **Vegetarian**.\n- **Experience level:** **Beginner** (workouts and habit changes should start easy and progress gradually).\n- **Physical considerations:** **Mild lower-back discomfort** (low-impact, back-friendly movements; prioritize form and core/glute strength).\n- **Meal preference:** **Simple meals** (minimal ingredients, low prep complexity).\n- **Workout preference:** **Home workouts** (no gym required; minimal equipment).\n- **Sleep preference:** **Bedtime at 10:30pm** (plan should support a consistent wind-down routine).\n\n## Exercise Plan\n\n### Weekly Schedule\n**Day 1: Monday**\n- **Activity**: 20-minute walk  \n- **Stretching**: 10 minutes (focus on lower back; include Cat-Cow and Bird Dog)\n\n**Day 2: Tuesday**\n- **Activity**: 15 minutes bodyweight exercises\n  - Squats (10–15 reps)\n  - Push-ups (knee or wall if needed) (5–10 reps)\n  - Planks (10–20 seconds)\n\n**Day 3: Wednesday**\n- **Activity**: Rest or gentle yoga (focus on flexibility and relaxation)\n\n**Day 4: Thursday**\n- **Activity**: 20-minute walk  \n- **Stretching**: 10 minutes (focus on lower back; include Cat-Cow and Bird Dog)\n\n**Day 5: Friday**\n- **Activity**: 15 minutes bodyweight exercises\n  - Lunges\n  - Wall sits\n  - Glute bridges\n\n**Day 6: Saturday**\n- **Activity**: 30-minute recreational activity (choose something enjoyable like swimming, cycling, or hiking)\n\n**Day 7: Sunday**\n- **Activity**: Rest (or light stretching if desired)\n\n### Gentle Exercises for Lower Back Discomfort\n- **Cat-Cow Stretch**: 10–15 repetitions  \n- **Bird Dog**: 10 repetitions per side, hold ~5 seconds\n\n### Tips\n- Keep movements **low-impact** and **pain-free**; prioritize good form over intensity.\n- Add difficulty gradually (slightly longer walks, 1–2 extra reps, or longer plank holds over time).\n- Hydrate throughout the day.\n- Aim to finish eating **2–3 hours before bedtime** (10:30 PM).\n\n## Nutrition Plan\n\n### Meal Plan (Simple Vegetarian Options)\n**Breakfast Options:**\n- Overnight oats with fruits and nuts\n- Smoothie with spinach, banana, and almond milk\n\n**Lunch Options:**\n- Quinoa salad with mixed vegetables and chickpeas\n- Vegetable stir-fry with tofu and brown rice\n\n**Dinner Options:**\n- Lentil soup with whole-grain bread\n- Stuffed bell peppers with brown rice and black beans\n\n**Snacks:**\n- Fresh fruits (apples, bananas, berries)\n- Carrot sticks with hummus\n- Yogurt (if included in your vegetarian preference)\n\n### Sample Day of Eating (Optional Template)\n- **Breakfast**: Overnight oats with berries and nuts  \n- **Mid-Morning Snack**: Apple with almond butter  \n- **Lunch**: Quinoa salad with mixed greens, cherry tomatoes, cucumber, olive oil dressing  \n- **Afternoon Snack**: Carrot sticks with hummus  \n- **Dinner**: Stir-fried tofu with mixed vegetables (broccoli, bell peppers, carrots) over brown rice  \n- **Evening**: Herbal tea  \n\n### Meal Prep Tips\n1. **Plan ahead**: Choose 3–4 main dishes for the week and batch-cook.\n2. **Keep healthy snacks ready**: Fruit, nuts, yogurt, hummus + cut veggies.\n3. **Hydration**: Drink plenty of water; keep a bottle visible.\n\n## Stress Reduction Techniques\n- **Deep breathing**: Inhale 4 counts, hold 4, exhale 4.\n- **Progressive muscle relaxation**: Tense/release muscle groups from toes to head.\n- **Grounding (5-4-3-2-1)**: 5 see, 4 hear, 3 feel, 2 smell, 1 taste.\n- **Nature walks**: Spend time outdoors to reduce stress.\n- **Mindfulness practice**: 5–10 minutes daily (meditation or mindful breathing).\n\n## Stress Management Plan\n\n### Daily Practices (30 minutes/day)\n#### Morning Routine (10 minutes)\n- **Mindfulness meditation / deep breathing**: 5 minutes\n- **Gentle stretching (lower back focus)**:\n  - Cat-Cow: 10–15 reps\n  - Bird Dog: 10 reps/side (hold ~5 seconds)\n\n#### Midday Activity (10 minutes)\n- **Home workout (beginner circuit)**:\n  - Squats: 10–15 reps\n  - Push-ups (knee or wall): 5–10 reps\n  - Plank: 10–20 seconds\n\n#### Evening Wind Down (10 minutes)\n- **Light walk**: 10 minutes *or*\n- **Evening stretching**: lower-back-focused gentle stretches\n\n### Additional Tips\n- **Social connection**: Regular check-ins with friends/family for emotional support.\n- **Limit screen time**: Especially close to bedtime; reduce news/social media.\n\n## Sleep Plan\n\n### Target Schedule\n- **Wind down by**: 10:00 PM  \n- **Lights out / bedtime**: 10:30 PM  \n\n### Sleep Hygiene\n- Keep a consistent sleep/wake schedule.\n- Create a calming pre-bed routine (reading, gentle stretching, soothing music).\n- Reduce screens before bed.\n- Keep the bedroom cool, dark, and quiet when possible.\n\n### 7-Day Wellness Plan for Weight Loss and Stress Reduction\n| Day | Activity | Meal Idea |\n|---|---|---|\n| **Monday** | 20-minute walk + 10 minutes Cat-Cow + Bird Dog | Vegetable stir-fry with tofu |\n| **Tuesday** | 15 minutes bodyweight (squats, push-ups, planks) | Quinoa salad with chickpeas |\n| **Wednesday** | Rest day or gentle yoga | Lentil soup + whole grain bread |\n| **Thursday** | 20-minute walk + 10 minutes Cat-Cow + Bird Dog | Stuffed bell peppers with rice |\n| **Friday** | 15 minutes bodyweight exercises | Vegetable-based simple meal (e.g., tofu + veg + grain) |\n| **Saturday** | 30-minute recreational activity | Vegetable curry + brown rice |\n| **Sunday** | Rest day or light stretching | Smoothie bowl with fruits and nuts |\n\n## Weekly Check-in Template\n- [ ] Completed workouts: _/5\n- [ ] Followed nutrition plan: _/7 days\n- [ ] Sleep routine followed: _/7 nights\n- [ ] Stress practice done: _/7 days\n\n## Disclaimer\nThis plan is informational and not medical advice. If you have medical conditions or pain, consult a clinician.'}
Generate plan #2 (sleep + energy)
req2 = """Create a wellness plan for someone who wants to improve sleep quality and daytime energy.
Constraints: wakes up 6:30am, screen-heavy job, no caffeine after 12pm, 20 minutes/day.
Preferences: short routines, light exercise, simple breakfast ideas.
"""

planner_graph.invoke({
    "messages": [HumanMessage(content=req2)],
    "filename": "wellness_plan_sleep_energy.md"
})
{'messages': [HumanMessage(content='Create a wellness plan for someone who wants to improve sleep quality and daytime energy.\nConstraints: wakes up 6:30am, screen-heavy job, no caffeine after 12pm, 20 minutes/day.\nPreferences: short routines, light exercise, simple breakfast ideas.\n', additional_kwargs={}, response_metadata={}, id='976b47ba-9f20-4f05-b3d9-1426c83e6c50'),
  AIMessage(content="The wellness plan has been successfully saved as 'wellness_plan_sleep_energy.md'. If you need any further assistance, feel free to ask!", additional_kwargs={}, response_metadata={}, id='fcb744fd-9022-4ab4-8954-7aa6c22b3004', tool_calls=[], invalid_tool_calls=[])],
 'filename': 'wellness_plan_sleep_energy.md',
 'goals_md': '## Goals\n- Improve overall sleep quality (fall asleep easier, fewer night awakenings, feel more rested on waking).\n- Increase daytime energy and reduce mid-afternoon fatigue.\n- Build consistent, sustainable daily habits that fit within **20 minutes/day**.\n- Support alertness and comfort for a **screen-heavy job** (reduce eye strain and mental fatigue).\n\n## Constraints & Preferences\n- **Wake time:** 6:30am (plan should support a consistent schedule around this).\n- **Work context:** Screen-heavy job (include strategies that are screen-compatible and reduce strain).\n- **Caffeine rule:** No caffeine after **12:00pm**.\n- **Time available:** **20 minutes per day** total for wellness activities.\n- **Preferences:**\n  - Short routines (minimal steps, easy to repeat).\n  - Light exercise (low-impact, accessible).\n  - Simple breakfast ideas (quick, minimal prep).',
 'exercise_md': '## Exercise Plan\n\n### Daily Schedule\n- **Wake Up:** 6:30 AM\n- **Exercise Duration:** 20 minutes\n- **Caffeine Cut-off:** 12 PM\n- **Screen Time Management:** Limit screens 1 hour before bed\n\n### 7-Day Plan\n\n| Day       | Morning Routine (20 min)                          | Breakfast Ideas                          |\n|-----------|--------------------------------------------------|-----------------------------------------|\n| **Monday**    | 10 min walk + 10 min stretching                  | Overnight oats with berries and nuts    |\n| **Tuesday**   | 15 min bodyweight exercises (squats, push-ups)  | Greek yogurt with honey and fruit       |\n| **Wednesday** | 20 min gentle yoga                               | Smoothie with spinach, banana, and almond milk |\n| **Thursday**  | 10 min walk + 10 min stretching                  | Whole grain toast with avocado          |\n| **Friday**    | 15 min bodyweight exercises (planks, lunges)    | Oatmeal with sliced banana and walnuts  |\n| **Saturday**  | 20 min recreational activity (cycling, hiking)   | Chia pudding with almond milk and fruit |\n| **Sunday**    | Rest or gentle yoga                              | Scrambled eggs with spinach and tomatoes |\n\n### Additional Tips\n- **Hydration:** Drink a glass of water first thing in the morning.\n- **Sleep Hygiene:** Maintain a cool, dark, and quiet bedroom. Consider blackout curtains and a sleep mask.\n- **Relaxing Bedtime Routine:** Engage in calming activities like reading or gentle stretching before bed.\n- **Energy Boosters:** Incorporate quick energy boosters during the day, such as brief walks or power poses. \n\nThis plan is designed to enhance sleep quality and boost daytime energy through light exercise and nutritious meals.',
 'nutrition_md': '## Nutrition Plan\n\n### Morning Routine (6:30 AM)\n- **Wake Up**: Start your day with a glass of water to hydrate.\n- **Breakfast Ideas**:\n  - Overnight oats topped with berries and nuts.\n  - Greek yogurt with honey and a sprinkle of granola.\n  - Smoothie with spinach, banana, and almond milk.\n\n### Daytime Energy Boosters\n- **Short Exercise (20 minutes)**:\n  - 5 minutes of stretching or yoga.\n  - 10 minutes of light cardio (e.g., brisk walking, jumping jacks).\n  - 5 minutes of deep breathing or meditation.\n\n- **Quick Energy Boosts**:\n  - Take a brief walk outside for sunlight.\n  - Drink a glass of cold water.\n  - Have a healthy snack like nuts or fruit.\n\n### Afternoon Routine\n- **Lunch**: Grilled chicken salad with mixed greens and olive oil dressing.\n- **Post-Lunch Snack**: Apple with almond butter or Greek yogurt.\n\n### Evening Routine\n- **Dinner**: Baked salmon with roasted vegetables and quinoa.\n- **Pre-Bedtime**: Herbal tea (e.g., chamomile) to promote relaxation.\n\n### Sleep Hygiene Tips\n- Maintain a consistent sleep schedule.\n- Limit screen time 1-2 hours before bed.\n- Create a relaxing bedtime routine (reading, gentle stretching).\n- Keep your bedroom cool, dark, and quiet (ideal temperature: 65-68°F).\n\n### Additional Tips\n- Avoid caffeine after 12 PM.\n- Stay hydrated throughout the day.\n- Incorporate relaxation techniques like progressive muscle relaxation or meditation before bed.',
 'sleep_md': '## Sleep Plan\n\n### Daily Schedule\n- **Wake Up:** 6:30 AM\n- **Bedtime:** Aim for 10:30 PM for 8 hours of sleep.\n\n### Morning Routine (6:30 AM - 7:00 AM)\n- **Hydrate:** Start with a glass of water.\n- **Light Exercise (10 minutes):** \n  - 5 minutes of stretching (focus on neck, shoulders, and back).\n  - 5 minutes of light cardio (e.g., jumping jacks or brisk walking).\n- **Breakfast Ideas (15 minutes):**\n  - Overnight oats with fruits.\n  - Greek yogurt with honey and nuts.\n  - Smoothie with spinach, banana, and almond milk.\n\n### Work Hours (7:00 AM - 5:00 PM)\n- **Screen Breaks:** Every hour, take a 5-minute break to stretch or walk.\n- **Caffeine:** Limit to before 12 PM. Opt for herbal teas post-lunch.\n\n### Afternoon Energy Boost (3:00 PM)\n- **Quick Energy Boost (5 minutes):**\n  - Stand up and do 10 jumping jacks.\n  - Drink a glass of cold water.\n  - Step outside for sunlight.\n\n### Evening Routine (5:00 PM - 10:30 PM)\n- **Dinner:** Light meal, avoid heavy foods 2-3 hours before bed.\n- **Relaxation (8:30 PM):**\n  - Engage in a calming activity (reading, gentle yoga, or meditation).\n  - Limit screen time; consider using blue light filters if necessary.\n- **Prepare for Sleep (10:00 PM):**\n  - Create a relaxing bedtime routine (warm bath, deep breathing).\n  - Ensure bedroom is cool (65-68°F), dark, and quiet.\n\n### Additional Tips\n- **Consistency:** Stick to the same sleep schedule, even on weekends.\n- **Environment:** Use blackout curtains and consider a white noise machine if needed.\n- **Hydration:** Stay hydrated throughout the day, but limit fluids close to bedtime.\n\nBy following this plan, you can improve your sleep quality and enhance your daytime energy levels.',
 'stress_md': '## Stress Management Plan\n\n### Morning Routine (6:30 AM - 7:00 AM)\n- **Wake Up**: Get out of bed immediately to avoid snoozing.\n- **Hydrate**: Drink a glass of water to kickstart your metabolism.\n- **Light Exercise (10 minutes)**: \n  - Stretching or yoga to wake up your body.\n  - Simple exercises like jumping jacks or a brisk walk around the house.\n- **Breakfast (10 minutes)**: \n  - Options: Greek yogurt with fruit, oatmeal with nuts, or a smoothie with spinach and banana.\n\n### Work Routine (Throughout the Day)\n- **Screen Breaks**: Every hour, take a 5-minute break to stand, stretch, or walk around.\n- **Mindfulness Practice (5 minutes)**: \n  - Mid-morning, practice deep breathing or a short meditation to refocus.\n  \n### Afternoon Routine (Post-Work)\n- **Light Exercise (10 minutes)**: \n  - Go for a walk outside or do a quick home workout to boost energy.\n  \n### Evening Routine (Before Bed)\n- **Dinner**: Eat a light meal, avoiding heavy foods and alcohol.\n- **Wind Down (1 hour before bed)**:\n  - Limit screen time; read a book or listen to calming music.\n  - Engage in a relaxing activity like gentle stretching or a warm bath.\n  \n### Sleep Hygiene\n- **Consistent Sleep Schedule**: Aim to go to bed at the same time each night.\n- **Bedroom Environment**: Keep your room cool, dark, and quiet.\n- **Limit Caffeine**: No caffeine after 12 PM to ensure better sleep quality.\n\nBy following this plan, you can improve your sleep quality and increase your daytime energy effectively.',
 'final_plan_md': '# Personal Wellness Plan\nGenerated: 2026-02-07\n\n## Goals\n- Improve overall sleep quality (fall asleep easier, fewer night awakenings, feel more rested on waking).\n- Increase daytime energy and reduce mid-afternoon fatigue.\n- Build consistent, sustainable daily habits that fit within **20 minutes/day**.\n- Support alertness and comfort for a **screen-heavy job** (reduce eye strain and mental fatigue).\n\n## Constraints & Preferences\n- **Wake time:** 6:30am (plan should support a consistent schedule around this).\n- **Work context:** Screen-heavy job (include strategies that are screen-compatible and reduce strain).\n- **Caffeine rule:** No caffeine after **12:00pm**.\n- **Time available:** **20 minutes per day** total for wellness activities.\n- **Preferences:**\n  - Short routines (minimal steps, easy to repeat).\n  - Light exercise (low-impact, accessible).\n  - Simple breakfast ideas (quick, minimal prep).\n\n## Exercise Plan\n\n### Daily Schedule\n- **Wake Up:** 6:30 AM\n- **Exercise Duration:** 20 minutes\n- **Caffeine Cut-off:** 12 PM\n- **Screen Time Management:** Limit screens 1 hour before bed\n- **Bedtime Target:** 10:30 PM (aiming for ~8 hours)\n\n### 7-Day Plan\n\n| Day       | Morning Routine (20 min)                          | Breakfast Ideas                          |\n|-----------|--------------------------------------------------|-----------------------------------------|\n| **Monday**    | 10 min walk + 10 min stretching                  | Overnight oats with berries and nuts    |\n| **Tuesday**   | 15 min bodyweight exercises (squats, push-ups)  | Greek yogurt with honey and fruit       |\n| **Wednesday** | 20 min gentle yoga                               | Smoothie with spinach, banana, and almond milk |\n| **Thursday**  | 10 min walk + 10 min stretching                  | Whole grain toast with avocado          |\n| **Friday**    | 15 min bodyweight exercises (planks, lunges)    | Oatmeal with sliced banana and walnuts  |\n| **Saturday**  | 20 min recreational activity (cycling, hiking)   | Chia pudding with almond milk and fruit |\n| **Sunday**    | Rest or gentle yoga                              | Scrambled eggs with spinach and tomatoes |\n\n### Additional Tips\n- **Hydration:** Drink a glass of water first thing in the morning.\n- **Screen-heavy work support:** Use brief eye breaks during the day (look far away, blink intentionally) and adjust brightness/contrast for comfort.\n- **Energy boosters (no extra “wellness time” required):** Brief standing/walking breaks, posture reset, quick sunlight exposure.\n- **Sleep hygiene basics:** Keep bedroom cool, dark, quiet (ideal ~65–68°F). Consider blackout curtains, sleep mask, or white noise.\n- **Relaxing bedtime routine:** Read, gentle stretching, calming music, or a warm bath during the last hour before bed.\n\n## Nutrition Plan\n\n### Morning Routine (6:30 AM)\n- **Wake Up:** Start with a glass of water to hydrate.\n- **Simple Breakfast Ideas (quick/minimal prep):**\n  - Overnight oats topped with berries and nuts.\n  - Greek yogurt with honey and a sprinkle of granola.\n  - Smoothie with spinach, banana, and almond milk.\n  - Whole grain toast with avocado.\n  - Oatmeal with banana and walnuts.\n  - Scrambled eggs with spinach and tomatoes.\n\n### Daytime Energy Boosters (screen-compatible, low effort)\n- **Movement snacks (1–5 minutes as needed):** brief walk, light stretching, shoulder/neck rolls, posture reset.\n- **Quick boosts:** sunlight break, cold water, protein/fiber snack (nuts, fruit, yogurt).\n- **Caffeine rule:** caffeine before **12 PM** only; consider herbal tea after lunch.\n\n### Afternoon Routine\n- **Lunch:** Grilled chicken salad with mixed greens and olive oil dressing.\n- **Post-Lunch Snack:** Apple with almond butter or Greek yogurt.\n\n### Evening Routine\n- **Dinner:** Baked salmon with roasted vegetables and quinoa (or similar balanced plate).\n- **Pre-Bedtime:** Herbal tea (e.g., chamomile) if desired; avoid heavy meals 2–3 hours before bed.\n\n### Additional Tips\n- Stay hydrated throughout the day, but consider limiting large fluid intake right before bed.\n- Aim for balanced meals (protein + fiber + healthy fats) to reduce energy crashes.\n\n## Stress Management Plan\n\n### Morning Routine (6:30 AM - 7:00 AM)\n- **Get up promptly:** Avoid extended snoozing to support circadian consistency.\n- **Hydrate:** 1 glass of water.\n- **Light movement:** Use your scheduled **20 minutes** (walk/stretch/yoga/bodyweight) to reduce stress and boost alertness.\n- **Breakfast:** Choose one quick option from the list above.\n\n### Work Routine (Throughout the Day)\n- **Screen breaks:** Every hour, take a 1–5 minute break to stand, stretch, and rest eyes.\n- **Mini-mindfulness (~1–5 minutes):** Deep breathing or a short reset mid-morning (keep it brief and repeatable).\n\n### Afternoon Routine (Post-Work)\n- If energy dips, use a short walk or gentle movement break (keep within the day’s overall time budget when possible).\n\n### Evening Routine (Before Bed)\n- **Dinner:** Keep it lighter; avoid heavy foods and alcohol close to bedtime.\n- **Wind down (1 hour before bed):** Reduce screens; choose calming activities (reading, gentle stretching, warm bath, calming music).\n\n### Sleep Hygiene\n- Keep bedtime/wake time consistent (target **10:30 PM** bedtime, **6:30 AM** wake time).\n- No caffeine after **12 PM**.\n- Bedroom: cool, dark, quiet.\n\n## Sleep Plan\n\n### Daily Schedule\n- **Wake Up:** 6:30 AM  \n- **Bedtime:** Aim for **10:30 PM**\n\n### Morning Routine (6:30 AM - 7:00 AM)\n- **Hydrate:** Glass of water.\n- **Movement (part of your 20 minutes/day):** Gentle stretching/walk/yoga/bodyweight depending on the day.\n- **Breakfast:** One simple option (overnight oats, yogurt, smoothie, toast, eggs).\n\n### Work Hours (Approx. 7:00 AM - 5:00 PM)\n- **Screen breaks:** Brief break at least once per hour (stand, stretch, look into the distance).\n- **Caffeine:** Before **12 PM** only.\n\n### Afternoon Energy Boost (Around 3:00 PM)\n- **5-minute reset (as needed):**\n  - Stand and move (e.g., a short walk or a few bodyweight reps).\n  - Drink cold water.\n  - Step outside for sunlight.\n\n### Evening Routine (5:00 PM - 10:30 PM)\n- **Dinner:** Finish heavier foods earlier; aim to stop large meals 2–3 hours before bed.\n- **Relaxation (around 8:30 PM):** Calming activity; reduce mental stimulation.\n- **Prepare for sleep (around 10:00 PM):** Low light, consistent routine, bedroom set to cool/dark/quiet.\n\n### Additional Tips\n- Keep weekends close to the same schedule where possible.\n- Consider blackout curtains, sleep mask, or white noise if awakenings are frequent.\n\n## Weekly Check-in Template\n- [ ] Completed workouts: _/5\n- [ ] Followed nutrition plan: _/7 days\n- [ ] Sleep routine followed: _/7 nights\n- [ ] Stress practice done: _/7 days\n\n## Disclaimer\nThis plan is informational and not medical advice. If you have medical conditions or pain, consult a clinician.'}
Verify the two files exist in plans/
print(list_saved_plans.invoke({}))
Saved plans:
- plan_demo_1.md
- plan_example_1.md
- plan_example_2.md
- wellness_plan_sleep_energy.md
- wellness_plan_weight_loss_stress.md
Summary
In this session, we:

Understood Multi-Agent Systems: When to use them and key patterns
Built a Supervisor Pattern: Central orchestrator routing to specialists
Implemented Agent Handoffs: Agents transferring control to each other
Added Web Search: Tavily for current information alongside knowledge base
Applied Context Engineering: Managing context for optimal performance
Key Takeaways:
Don't over-engineer: Only add agents when you truly need specialization
Context is key: Manage your context window carefully
Patterns matter: Choose the right pattern for your use case
Further Reading:

Building Effective Agents (Anthropic)
Don't Build Multi-Agents (Cognition)
12-Factor Agents