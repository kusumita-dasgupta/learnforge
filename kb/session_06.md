
Agent Memory: Building Memory-Enabled Agents with LangGraph
In this notebook, we'll explore agent memory systems - the ability for AI agents to remember information across interactions. We'll implement all five memory types from the CoALA (Cognitive Architectures for Language Agents) framework while building on our Personal Wellness Assistant use case.

Learning Objectives:

Understand the 5 memory types from the CoALA framework
Implement short-term memory with checkpointers and thread_id
Build long-term memory with InMemoryStore and namespaces
Use semantic memory for meaning-based retrieval
Apply episodic memory for few-shot learning from past experiences
Create procedural memory for self-improving agents
Combine all memory types into a unified wellness agent
Table of Contents:
Breakout Room #1: Memory Foundations

Task 1: Dependencies
Task 2: Understanding Agent Memory (CoALA Framework)
Task 3: Short-Term Memory (MemorySaver, thread_id)
Task 4: Long-Term Memory (InMemoryStore, namespaces)
Task 5: Message Trimming & Context Management
Question #1 & Question #2
🏗️ Activity #1: Store & Retrieve User Wellness Profile
Breakout Room #2: Advanced Memory & Integration

Task 6: Semantic Memory (Embeddings + Search)
Task 7: Building Semantic Wellness Knowledge Base
Task 8: Episodic Memory (Few-Shot Learning)
Task 9: Procedural Memory (Self-Improving Agent)
Task 10: Unified Wellness Memory Agent
Question #3 & Question #4
🏗️ Activity #2: Wellness Memory Dashboard
🤝 Breakout Room #1
Memory Foundations
Task 1: Dependencies
Before we begin, make sure you have:

API Keys for:

OpenAI (for GPT-4o-mini and embeddings)
LangSmith (optional, for tracing)
Dependencies installed via uv sync

# Core imports
import os
import getpass
from uuid import uuid4
from typing import Annotated, TypedDict

import nest_asyncio
nest_asyncio.apply()  # Required for async operations in Jupyter
# Set API Keys
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
# Optional: LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE9 - Agent Memory - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key (press Enter to skip): ") or ""

if not os.environ["LANGCHAIN_API_KEY"]:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled")
else:
    print(f"LangSmith tracing enabled. Project: {os.environ['LANGCHAIN_PROJECT']}")
LangSmith tracing enabled. Project: AIE9 - Agent Memory - d8c6ef48
# Initialize LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Test the connection
response = llm.invoke("Say 'Memory systems ready!' in exactly those words.")
print(response.content)
Memory systems ready!
Task 2: Understanding Agent Memory (CoALA Framework)
The CoALA (Cognitive Architectures for Language Agents) framework identifies 5 types of memory that agents can use:

Memory Type	Human Analogy	AI Implementation	Wellness Example
Short-term	What someone just said	Conversation history within a thread	Current consultation conversation
Long-term	Remembering a friend's birthday	User preferences stored across sessions	User's goals, allergies, conditions
Semantic	Knowing Paris is in France	Facts retrieved by meaning	Wellness knowledge retrieval
Episodic	Remembering your first day at work	Learning from past experiences	Past successful advice patterns
Procedural	Knowing how to ride a bike	Self-improving instructions	Learned communication preferences
Memory Architecture Overview
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Wellness Agent                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Short-term  │  │  Long-term   │  │   Semantic   │           │
│  │    Memory    │  │    Memory    │  │    Memory    │           │
│  │              │  │              │  │              │           │
│  │ Checkpointer │  │    Store     │  │Store+Embed   │           │
│  │ + thread_id  │  │ + namespace  │  │  + search()  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   Episodic   │  │  Procedural  │                             │
│  │    Memory    │  │    Memory    │                             │
│  │              │  │              │                             │
│  │  Few-shot    │  │Self-modifying│                             │
│  │  examples    │  │   prompts    │                             │
│  └──────────────┘  └──────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
Key LangGraph Components
Component	Memory Type	Scope
MemorySaver (Checkpointer)	Short-term	Within a single thread
InMemoryStore	Long-term, Semantic, Episodic, Procedural	Across all threads
thread_id	Short-term	Identifies unique conversations
Namespaces	All store-based	Organizes memories by user/purpose
Documentation:

CoALA Paper
LangGraph Memory Concepts
Task 3: Short-Term Memory (MemorySaver, thread_id)
Short-term memory maintains context within a single conversation thread. Think of it like your working memory during a phone call - you remember what was said earlier, but once the call ends, those details fade.

In LangGraph, short-term memory is implemented through:

Checkpointer: Saves the graph state at each step
thread_id: Uniquely identifies each conversation
How It Works
Thread 1: "Hi, I'm Alice"          Thread 2: "What's my name?"
     │                                   │
     ▼                                   ▼
┌──────────────┐                   ┌──────────────┐
│ Checkpointer │                   │ Checkpointer │
│  thread_1    │                   │  thread_2    │
│              │                   │              │
│ ["Hi Alice"] │                   │ [empty]      │
└──────────────┘                   └──────────────┘
     │                                   │
     ▼                                   ▼
"Hi Alice!"                        "I don't know your name"
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Define the state schema for our graph
# The `add_messages` annotation tells LangGraph how to update the messages list
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define our wellness chatbot node
def wellness_chatbot(state: State):
    """Process the conversation and generate a wellness-focused response."""
    system_prompt = SystemMessage(content="""You are a friendly Personal Wellness Assistant. 
Help users with exercise, nutrition, sleep, and stress management questions.
Be supportive and remember details the user shares about themselves.""")
    
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the graph
builder = StateGraph(State)
builder.add_node("chatbot", wellness_chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile with a checkpointer for short-term memory
checkpointer = MemorySaver()
wellness_graph = builder.compile(checkpointer=checkpointer)

print("Wellness chatbot compiled with short-term memory (checkpointing)")
Wellness chatbot compiled with short-term memory (checkpointing)
# Test short-term memory within a thread
config = {"configurable": {"thread_id": "wellness_thread_1"}}

# First message - introduce ourselves
response = wellness_graph.invoke(
    {"messages": [HumanMessage(content="Hi! My name is Sarah and I want to improve my sleep.")]},
    config
)
print("User: Hi! My name is Sarah and I want to improve my sleep.")
print(f"Assistant: {response['messages'][-1].content}")
print()
User: Hi! My name is Sarah and I want to improve my sleep.
Assistant: Hi Sarah! It's great to meet you, and I'm glad you're focusing on improving your sleep. Sleep is so important for overall wellness. Can you tell me a bit more about your current sleep habits? For example, how many hours do you usually get, and do you have any trouble falling or staying asleep?

# Second message - test if it remembers (same thread)
response = wellness_graph.invoke(
    {"messages": [HumanMessage(content="What's my name and what am I trying to improve?")]},
    config  # Same config = same thread_id
)
print("User: What's my name and what am I trying to improve?")
print(f"Assistant: {response['messages'][-1].content}")
User: What's my name and what am I trying to improve?
Assistant: Your name is Sarah, and you're trying to improve your sleep. If you have any specific questions or areas you'd like to focus on regarding your sleep, just let me know!
# New thread - it won't remember Sarah!
different_config = {"configurable": {"thread_id": "wellness_thread_2"}}

response = wellness_graph.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    different_config  # Different thread_id = no memory of Sarah
)
print("User (NEW thread): What's my name?")
print(f"Assistant: {response['messages'][-1].content}")
print()
print("Notice: The agent doesn't know our name because this is a new thread!")
User (NEW thread): What's my name?
Assistant: I don't have your name yet! If you'd like to share it, I can remember it for our future conversations. How can I assist you today?

Notice: The agent doesn't know our name because this is a new thread!
# Inspect the state of thread 1
state = wellness_graph.get_state(config)
print(f"Thread 1 has {len(state.values['messages'])} messages:")
for msg in state.values['messages']:
    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
    content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    print(f"  {role}: {content}")
Thread 1 has 6 messages:
  User: Hi! My name is Sarah and I want to improve my sleep.
  Assistant: Hi Sarah! It's great to meet you, and I'm glad you're focusing on improving your...
  User: What's my name and what am I trying to improve?
  Assistant: Your name is Sarah, and you're trying to improve your sleep. If you have any spe...
  User: What's my name and what am I trying to improve?
  Assistant: Your name is Sarah, and you're trying to improve your sleep. If you have any spe...
Task 4: Long-Term Memory (InMemoryStore, namespaces)
Long-term memory stores information across different conversation threads. This is like remembering that your friend prefers tea over coffee - you remember it every time you meet them, regardless of what you're currently discussing.

In LangGraph, long-term memory uses:

Store: A persistent key-value store
Namespaces: Organize memories by user, application, or context
Key Difference from Short-Term Memory
Short-Term (Checkpointer)	Long-Term (Store)
Scoped to a single thread	Shared across all threads
Automatic (messages)	Explicit (you decide what to store)
Conversation history	User preferences, facts, etc.
from langgraph.store.memory import InMemoryStore

# Create a store for long-term memory
store = InMemoryStore()

# Namespaces organize memories - typically by user_id and category
user_id = "user_sarah"
profile_namespace = (user_id, "profile")
preferences_namespace = (user_id, "preferences")

# Store Sarah's wellness profile
store.put(profile_namespace, "name", {"value": "Sarah"})
store.put(profile_namespace, "goals", {"primary": "improve sleep", "secondary": "reduce stress"})
store.put(profile_namespace, "conditions", {"allergies": ["peanuts"], "injuries": ["bad knee"]})

# Store Sarah's preferences
store.put(preferences_namespace, "communication", {"style": "friendly", "detail_level": "moderate"})
store.put(preferences_namespace, "schedule", {"preferred_workout_time": "morning", "available_days": ["Mon", "Wed", "Fri"]})

print("Stored Sarah's profile and preferences in long-term memory")
Stored Sarah's profile and preferences in long-term memory
# Retrieve specific memories
name = store.get(profile_namespace, "name")
print(f"Name: {name.value}")

goals = store.get(profile_namespace, "goals")
print(f"Goals: {goals.value}")

# List all memories in a namespace
print("\nAll profile items:")
for item in store.search(profile_namespace):
    print(f"  {item.key}: {item.value}")
Name: {'value': 'Sarah'}
Goals: {'primary': 'improve sleep', 'secondary': 'reduce stress'}

All profile items:
  name: {'value': 'Sarah'}
  goals: {'primary': 'improve sleep', 'secondary': 'reduce stress'}
  conditions: {'allergies': ['peanuts'], 'injuries': ['bad knee']}
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

# Define state with user_id for personalization
class PersonalizedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def personalized_wellness_chatbot(state: PersonalizedState, config: RunnableConfig, *, store: BaseStore):
    """A wellness chatbot that uses long-term memory for personalization."""
    user_id = state["user_id"]
    profile_namespace = (user_id, "profile")
    preferences_namespace = (user_id, "preferences")
    
    # Retrieve user profile from long-term memory
    profile_items = list(store.search(profile_namespace))
    pref_items = list(store.search(preferences_namespace))
    
    # Build context from profile
    profile_text = "\n".join([f"- {p.key}: {p.value}" for p in profile_items])
    pref_text = "\n".join([f"- {p.key}: {p.value}" for p in pref_items])
    
    system_msg = f"""You are a Personal Wellness Assistant. You know the following about this user:

PROFILE:
{profile_text if profile_text else 'No profile stored.'}

PREFERENCES:
{pref_text if pref_text else 'No preferences stored.'}

Use this information to personalize your responses. Be supportive and helpful."""
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the personalized graph
builder2 = StateGraph(PersonalizedState)
builder2.add_node("chatbot", personalized_wellness_chatbot)
builder2.add_edge(START, "chatbot")
builder2.add_edge("chatbot", END)

# Compile with BOTH checkpointer (short-term) AND store (long-term)
personalized_graph = builder2.compile(
    checkpointer=MemorySaver(),
    store=store
)

print("Personalized graph compiled with both short-term and long-term memory")
Personalized graph compiled with both short-term and long-term memory
# Test the personalized chatbot - it knows Sarah's profile!
config = {"configurable": {"thread_id": "personalized_thread_1"}}

response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="What exercises would you recommend for me?")],
        "user_id": "user_sarah"
    },
    config
)

print("User: What exercises would you recommend for me?")
print(f"Assistant: {response['messages'][-1].content}")
print()
print("Notice: The agent knows about Sarah's bad knee without her mentioning it!")
User: What exercises would you recommend for me?
Assistant: Hi Sarah! It's great that you're looking to improve your sleep and reduce stress through exercise. Given your bad knee, we want to focus on low-impact activities that are gentle on your joints. Here are some exercises you might enjoy, especially in the morning:

1. **Walking**: A brisk walk is a fantastic way to start your day. It’s low-impact and can help clear your mind.

2. **Swimming**: If you have access to a pool, swimming is excellent for a full-body workout without stressing your knee.

3. **Cycling**: Riding a stationary bike or cycling outdoors can be a great way to get your heart rate up while being easy on your joints.

4. **Yoga**: Gentle yoga can help with flexibility, relaxation, and stress reduction. Look for classes that focus on restorative or yin yoga.

5. **Pilates**: This can strengthen your core and improve your overall body awareness without putting too much strain on your knee.

6. **Tai Chi**: This is a gentle form of martial arts that focuses on slow, controlled movements and can be very calming.

Make sure to listen to your body and modify any exercises as needed. It might also be helpful to consult with a fitness professional who can tailor a program specifically for you. Let me know if you’d like more details on any of these options!

Notice: The agent knows about Sarah's bad knee without her mentioning it!
# Even in a NEW thread, it still knows Sarah's profile
# because long-term memory is cross-thread!

new_config = {"configurable": {"thread_id": "personalized_thread_2"}}

response = personalized_graph.invoke(
    {
        "messages": [HumanMessage(content="Can you suggest a snack for me?")],
        "user_id": "user_sarah"
    },
    new_config
)

print("User (NEW thread): Can you suggest a snack for me?")
print(f"Assistant: {response['messages'][-1].content}")
print()
print("Notice: Even in a new thread, the agent knows Sarah has a peanut allergy!")
User (NEW thread): Can you suggest a snack for me?
Assistant: Absolutely, Sarah! Since you have a peanut allergy, let’s find something safe and delicious. How about some apple slices with almond butter? It’s a great source of healthy fats and protein, and the sweetness of the apple can be really satisfying. Just make sure to choose almond butter that’s made in a peanut-free facility to avoid any cross-contamination. 

If you’re looking for something lighter, you could also try Greek yogurt with some berries. It’s nutritious and can be a great way to help with your overall wellness goals. Let me know if you’d like more options!

Notice: Even in a new thread, the agent knows Sarah has a peanut allergy!
Task 5: Message Trimming & Context Management
Long conversations can exceed the LLM's context window. LangGraph provides utilities to manage message history:

trim_messages: Keeps only recent messages up to a token limit
Summarization: Compress older messages into summaries
Why Trim Even with 128K Context?
Even with large context windows:

Cost: More tokens = higher API costs
Latency: Larger contexts take longer to process
Quality: Models can struggle with "lost in the middle" - important info buried in long contexts
Relevance: Old messages may not be relevant to current query
from langchain_core.messages import trim_messages

# Create a trimmer that keeps only recent messages
trimmer = trim_messages(
    max_tokens=500,  # Keep messages up to this token count
    strategy="last",  # Keep the most recent messages
    token_counter=llm,  # Use the LLM to count tokens
    include_system=True,  # Always keep system messages
    allow_partial=False,  # Don't cut messages in half
)

# Example: Create a long conversation
long_conversation = [
    SystemMessage(content="You are a wellness assistant."),
    HumanMessage(content="I want to improve my health."),
    AIMessage(content="Great goal! Let's start with exercise. What's your current activity level?"),
    HumanMessage(content="I walk about 30 minutes a day."),
    AIMessage(content="That's a good foundation. For cardiovascular health, aim for 150 minutes of moderate activity per week."),
    HumanMessage(content="What about nutrition?"),
    AIMessage(content="Focus on whole foods: vegetables, lean proteins, whole grains. Limit processed foods and added sugars."),
    HumanMessage(content="And sleep?"),
    AIMessage(content="Aim for 7-9 hours. Maintain a consistent sleep schedule and create a relaxing bedtime routine."),
    HumanMessage(content="What's the most important change I should make first?"),
]

# Trim to fit context window
trimmed = trimmer.invoke(long_conversation)
print(f"Original: {len(long_conversation)} messages")
print(f"Trimmed: {len(trimmed)} messages")
print("\nTrimmed conversation:")
for msg in trimmed:
    role = type(msg).__name__.replace("Message", "")
    content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
    print(f"  {role}: {content}")
Original: 10 messages
Trimmed: 10 messages

Trimmed conversation:
  System: You are a wellness assistant.
  Human: I want to improve my health.
  AI: Great goal! Let's start with exercise. What's your current a...
  Human: I walk about 30 minutes a day.
  AI: That's a good foundation. For cardiovascular health, aim for...
  Human: What about nutrition?
  AI: Focus on whole foods: vegetables, lean proteins, whole grain...
  Human: And sleep?
  AI: Aim for 7-9 hours. Maintain a consistent sleep schedule and ...
  Human: What's the most important change I should make first?
# Summarization approach for longer conversations

def summarize_conversation(messages: list, max_messages: int = 6) -> list:
    """Summarize older messages to manage context length."""
    if len(messages) <= max_messages:
        return messages
    
    # Keep the system message and last few messages
    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    content_messages = messages[1:] if system_msg else messages
    
    if len(content_messages) <= max_messages:
        return messages
    
    old_messages = content_messages[:-max_messages+1]
    recent_messages = content_messages[-max_messages+1:]
    
    # Summarize old messages
    summary_prompt = f"""Summarize this conversation in 2-3 sentences, 
capturing key wellness topics discussed and any important user information:

{chr(10).join([f'{type(m).__name__}: {m.content[:200]}' for m in old_messages])}"""
    
    summary = llm.invoke(summary_prompt)
    
    # Return: system + summary + recent messages
    result = []
    if system_msg:
        result.append(system_msg)
    result.append(SystemMessage(content=f"[Previous conversation summary: {summary.content}]"))
    result.extend(recent_messages)
    
    return result


# Test summarization
summarized = summarize_conversation(long_conversation, max_messages=4)
print(f"Summarized: {len(summarized)} messages")
print("\nSummarized conversation:")
for msg in summarized:
    role = type(msg).__name__.replace("Message", "")
    content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    print(f"  {role}: {content}")
Summarized: 5 messages

Summarized conversation:
  System: You are a wellness assistant.
  System: [Previous conversation summary: The conversation centers around the user's desir...
  Human: And sleep?
  AI: Aim for 7-9 hours. Maintain a consistent sleep schedule and create a relaxing be...
  Human: What's the most important change I should make first?
❓ Question #1:
What are the trade-offs between short-term memory (checkpointer) vs long-term memory (store)? When should wellness data move from short-term to long-term?

Consider:

What information should persist across sessions?
What are the privacy implications of each?
How would you decide what to promote from short-term to long-term?
Answer:
Short-term memory (the checkpointer + thread_id) is basically “what we’re talking about right now.” It’s automatic, cheap to use, and great for continuity inside a single session, but it is session-scoped: once you start a new thread, that memory won’t follow you. It’s also messy by nature because it’s raw conversation content—useful for immediate context, but not structured, not curated, and easy to bloat. The upside is that it naturally expires when the thread ends, which is a privacy win.

Long-term memory (the store + namespaces) is “what should still be true tomorrow.” It persists across threads, is explicitly written and retrieved, and can be structured (profile, allergies, goals, preferences). The trade-off is that it creates data stewardship obligations: you now need rules for consent, retention, access control, deletion, and minimizing what you store. Long-term memory is more powerful for personalization and safety, but it’s also higher risk because it retains user information beyond the immediate conversation.

Wellness data should move from short-term to long-term only when it meets two criteria: (1) it will be useful across sessions, and (2) it is safe and appropriate to retain. In practice, that means persisting things like allergies, chronic conditions, injuries/limitations, goals, baseline preferences (communication style, dietary restrictions), and “hard constraints” that affect future recommendations. You generally should not persist highly sensitive, transient, or emotionally revealing details (e.g., a bad day, a fight with someone, specific symptoms that sound medical) unless the user explicitly wants it remembered and you have a clear purpose.

Privacy-wise, checkpointer memory is lower risk because it’s bounded to a thread and can be treated as ephemeral. Store-based memory is higher risk because it becomes a user profile over time. For a wellness assistant, you should treat long-term memory like PII/health-adjacent data: minimize it, separate it by namespaces, restrict access, support “forget me,” and consider TTL/expiration for anything that isn’t safety-critical.

To decide what to “promote” from short-term to long-term, use a promotion rule like: “stable + reusable + user-consented + low sensitivity”. Concretely: after a session, extract a small set of candidate facts (e.g., “peanut allergy,” “bad knee,” “goal: improve sleep,” “prefers morning workouts”), confirm them (“Do you want me to remember this for next time?”), then write only the confirmed, structured items into the store. Everything else stays in the thread history and eventually disappears.

❓ Question #2:
Why use message trimming with a 128K context window when HealthWellnessGuide.txt is only ~16KB? What should always be preserved when trimming a wellness consultation?

Consider:

The "lost in the middle" phenomenon
Cost and latency implications
What user information is critical for safety (allergies, conditions, etc.)
Answer:
Even with a 128K context window, message trimming is still necessary because capacity is not the same as usefulness. Large context windows can technically hold long conversations and documents, but models do not attend to all tokens equally. The “lost in the middle” effect means important details buried deep in long histories are more likely to be ignored or diluted, especially when newer instructions and retrieved knowledge compete for attention. In a wellness setting, this can cause the model to miss critical constraints or safety information even though it is technically present in the context.

Cost and latency are the second reason. Every extra token sent to the model increases both inference cost and response time. If you repeatedly include long conversational histories plus retrieved chunks—even if the knowledge base itself is only ~16KB—the total prompt can grow quickly across turns. Trimming keeps each turn efficient and predictable, which is essential for a production wellness assistant where conversations may span many interactions.

More importantly, trimming is about intentional preservation, not just reduction. In a wellness consultation, certain information must always survive trimming because it affects safety and correctness. This includes allergies, injuries, chronic conditions, contraindications, and core user goals. These should never rely on raw conversation history alone; they should either live in long-term memory or be explicitly re-injected into the system prompt. The conversation history itself can be shortened, but safety-critical facts must remain visible to the model on every turn.

In practice, trimming should preserve three things: (1) the system instructions and current procedural rules, (2) the most recent user–assistant exchanges that define the immediate question, and (3) any safety-critical or constraint information relevant to the user. Everything else—older chit-chat, exploratory questions, or resolved topics—can be summarized or dropped. This approach balances model attention, cost, and safety, ensuring the assistant remains responsive, accurate, and trustworthy even in long-running wellness conversations.

🏗️ Activity #1: Store & Retrieve User Wellness Profile
Build a complete wellness profile system that:

Defines a wellness profile schema (name, goals, conditions, preferences)
Creates functions to store and retrieve profile data
Builds a personalized wellness agent that uses the profile
Tests that different users get different advice
Requirements:
Define at least 5 profile attributes
Support multiple users with different profiles
Agent should reference profile data in responses
Step 1: Define a wellness profile schema Example attributes: name, age, goals, conditions, allergies, fitness_level, preferred_activities

### YOUR CODE HERE ###


from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage


# Step 1: Define a wellness profile schema
# Example attributes: name, age, goals, conditions, allergies, fitness_level, preferred_activities

class WellnessProfile(TypedDict, total=False):
    name: str
    age: int
    goals: list[str]
    conditions: list[str]
    allergies: list[str]
    injuries: list[str]
    fitness_level: str
    preferred_activities: list[str]
    dietary_preferences: list[str]
    communication_style: str
    schedule: dict


def _profile_ns(user_id: str) -> tuple:
    return (user_id, "profile")
Step 2: Create helper functions to store and retrieve profiles

# Step 2: Create helper functions to store and retrieve profiles
def store_wellness_profile(store: BaseStore, user_id: str, profile: dict):
    """Store a user's wellness profile."""
    if not user_id:
        raise ValueError("user_id is required")
    if not isinstance(profile, dict) or not profile:
        raise ValueError("profile must be a non-empty dict")

    # Minimal validation for safety-critical operation
    if not profile.get("name"):
        raise ValueError("profile must include 'name'")
    if "allergies" in profile and not isinstance(profile["allergies"], list):
        raise ValueError("'allergies' must be a list if provided")
    if "conditions" in profile and not isinstance(profile["conditions"], list):
        raise ValueError("'conditions' must be a list if provided")

    ns = _profile_ns(user_id)

    # Store each attribute separately for easy partial updates
    for key, value in profile.items():
        store.put(ns, key, {"value": value})


def get_wellness_profile(store: BaseStore, user_id: str) -> dict:
    """Retrieve a user's wellness profile."""
    ns = _profile_ns(user_id)
    items = list(store.search(ns))
    if not items:
        return {}

    profile = {}
    for item in items:
        profile[item.key] = item.value.get("value")
    return profile


def _format_profile(profile: dict) -> str:
    """Format profile into a concise system context string."""
    if not profile:
        return "No profile stored."

    # Put safety-critical items first
    lines = []
    lines.append(f"Name: {profile.get('name', 'Unknown')}")
    if "age" in profile:
        lines.append(f"Age: {profile.get('age')}")
    if profile.get("goals"):
        lines.append(f"Goals: {profile.get('goals')}")
    if profile.get("conditions"):
        lines.append(f"Conditions: {profile.get('conditions')}")
    if profile.get("injuries"):
        lines.append(f"Injuries/limitations: {profile.get('injuries')}")
    if profile.get("allergies"):
        lines.append(f"Allergies: {profile.get('allergies')}")
    if profile.get("fitness_level"):
        lines.append(f"Fitness level: {profile.get('fitness_level')}")
    if profile.get("preferred_activities"):
        lines.append(f"Preferred activities: {profile.get('preferred_activities')}")
    if profile.get("dietary_preferences"):
        lines.append(f"Dietary preferences: {profile.get('dietary_preferences')}")
    if profile.get("communication_style"):
        lines.append(f"Communication style: {profile.get('communication_style')}")
    if profile.get("schedule"):
        lines.append(f"Schedule: {profile.get('schedule')}")

    return "\n".join(lines)
Step 3: Create two different user profiles

# Step 3: Create two different user profiles

store = InMemoryStore()

profile_sarah: WellnessProfile = {
    "name": "Sarah",
    "age": 34,
    "goals": ["improve sleep", "reduce stress"],
    "conditions": ["mild anxiety"],
    "allergies": ["peanuts"],
    "injuries": ["bad knee"],
    "fitness_level": "beginner",
    "preferred_activities": ["walking", "yoga", "swimming"],
    "dietary_preferences": ["high-protein breakfast", "low added sugar"],
    "communication_style": "friendly",
    "schedule": {"workout_time": "morning", "days": ["Mon", "Wed", "Fri"]},
}

profile_mike: WellnessProfile = {
    "name": "Mike",
    "age": 45,
    "goals": ["lose weight", "improve cardio"],
    "conditions": ["hypertension"],
    "allergies": [],
    "injuries": ["shoulder impingement"],
    "fitness_level": "intermediate",
    "preferred_activities": ["cycling", "brisk walking"],
    "dietary_preferences": ["low sodium"],
    "communication_style": "concise",
    "schedule": {"workout_time": "evening", "days": ["Tue", "Thu", "Sat"]},
}

store_wellness_profile(store, "user_sarah", profile_sarah)
store_wellness_profile(store, "user_mike", profile_mike)

print("Stored profiles:", get_wellness_profile(store, "user_sarah").get("name"), "and", get_wellness_profile(store, "user_mike").get("name"))
Stored profiles: Sarah and Mike
Step 4: Build a personalized agent that uses profiles

# Step 4: Build a personalized agent that uses profiles

class ProfileState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def personalized_wellness_node(state: ProfileState, config: RunnableConfig, *, store: BaseStore):
    user_id = state["user_id"]
    profile = get_wellness_profile(store, user_id)
    profile_text = _format_profile(profile)

    system_msg = f"""You are a Personal Wellness Assistant.

Use the user's profile to personalize your advice.
- Always respect safety constraints (allergies, conditions, injuries).
- Match the user's communication style if provided (concise vs detailed).
- If a recommendation might conflict with conditions/injuries, provide safer alternatives.

USER PROFILE:
{profile_text}
"""
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(ProfileState)
builder.add_node("chatbot", personalized_wellness_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

profile_graph = builder.compile(
    checkpointer=MemorySaver(),  # short-term memory per thread
    store=store                 # long-term profile store across threads
)

print("Personalized profile agent compiled.")
Personalized profile agent compiled.
Step 5: Test with different users - they should get different advice

# Step 5: Test with different users - they should get different advice

q1 = "Suggest a simple workout plan for me this week."
resp_sarah = profile_graph.invoke(
    {"messages": [HumanMessage(content=q1)], "user_id": "user_sarah"},
    {"configurable": {"thread_id": "thread_sarah_1"}}
)
print("\n--- Sarah ---")
print(resp_sarah["messages"][-1].content)

resp_mike = profile_graph.invoke(
    {"messages": [HumanMessage(content=q1)], "user_id": "user_mike"},
    {"configurable": {"thread_id": "thread_mike_1"}}
)
print("\n--- Mike ---")
print(resp_mike["messages"][-1].content)

q2 = "Suggest a healthy snack."
resp_sarah2 = profile_graph.invoke(
    {"messages": [HumanMessage(content=q2)], "user_id": "user_sarah"},
    {"configurable": {"thread_id": "thread_sarah_2"}}
)
print("\n--- Sarah (snack) ---")
print(resp_sarah2["messages"][-1].content)

resp_mike2 = profile_graph.invoke(
    {"messages": [HumanMessage(content=q2)], "user_id": "user_mike"},
    {"configurable": {"thread_id": "thread_mike_2"}}
)
print("\n--- Mike (snack) ---")
print(resp_mike2["messages"][-1].content)
--- Sarah ---
Hi Sarah! Here’s a simple workout plan for you this week that focuses on walking, yoga, and swimming, keeping your knee in mind and aiming to reduce stress and improve sleep.

**Weekly Workout Plan:**

**Monday:**
- **Walking:** 20-30 minutes at a comfortable pace. Choose a flat route to be gentle on your knee.

**Tuesday:**
- **Yoga:** 30 minutes of gentle yoga focusing on relaxation and stretching. Look for beginner classes that emphasize restorative poses.

**Wednesday:**
- **Swimming:** 30 minutes of light swimming. This is great for your knee and provides a full-body workout without impact.

**Thursday:**
- **Walking:** 20-30 minutes again. Try to find a scenic route to make it more enjoyable!

**Friday:**
- **Yoga:** 30 minutes of yoga, focusing on deep breathing and relaxation techniques to help with anxiety.

**Saturday:**
- **Swimming:** 30 minutes, mixing in some light strokes and water exercises that don’t strain your knee.

**Sunday:**
- **Rest Day:** Take a break and focus on relaxation techniques, like meditation or deep breathing exercises.

Feel free to adjust the duration based on how you feel, and remember to listen to your body. Enjoy your workouts! 😊

--- Mike ---
Here's a simple workout plan for you this week:

**Monday:**  
- Brisk walking: 30 minutes

**Tuesday:**  
- Cycling: 45 minutes at a moderate pace

**Wednesday:**  
- Rest day or light stretching

**Thursday:**  
- Brisk walking: 30 minutes

**Friday:**  
- Cycling: 30 minutes with intervals (1 minute fast, 2 minutes slow)

**Saturday:**  
- Brisk walking: 30 minutes

**Sunday:**  
- Rest day or light stretching

Make sure to monitor your heart rate and stay within a safe range due to your hypertension. Enjoy your workouts!

--- Sarah (snack) ---
Hey Sarah! A great healthy snack for you could be Greek yogurt topped with some fresh berries. It’s high in protein and low in added sugar, plus the berries add a nice touch of sweetness and antioxidants. Just make sure to choose a yogurt that doesn’t have added sugars. Enjoy!

--- Mike (snack) ---
Try a small serving of unsalted almonds or a sliced apple with a bit of almond butter. Both are low in sodium and provide healthy fats and fiber.
🤝 Breakout Room #2
Advanced Memory & Integration
Task 6: Semantic Memory (Embeddings + Search)
Semantic memory stores facts and retrieves them based on meaning rather than exact matches. This is like how you might remember "that restaurant with the great pasta" even if you can't remember its exact name.

In LangGraph, semantic memory uses:

Store with embeddings: Converts text to vectors for similarity search
store.search(): Finds relevant memories by semantic similarity
How It Works
User asks: "What helps with headaches?"
         ↓
Query embedded → [0.2, 0.8, 0.1, ...]
         ↓
Compare with stored wellness facts:
  - "Hydration can relieve headaches" → 0.92 similarity ✓
  - "Exercise improves sleep" → 0.35 similarity
         ↓
Return: "Hydration can relieve headaches"
from langchain_openai import OpenAIEmbeddings

# Create embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a store with semantic search enabled
semantic_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # Dimension of text-embedding-3-small
    }
)

print("Semantic memory store created with embedding support")
Semantic memory store created with embedding support
# Store various wellness facts as semantic memories
namespace = ("wellness", "facts")

wellness_facts = [
    ("fact_1", {"text": "Drinking water can help relieve headaches caused by dehydration"}),
    ("fact_2", {"text": "Regular exercise improves sleep quality and helps you fall asleep faster"}),
    ("fact_3", {"text": "Deep breathing exercises can reduce stress and anxiety within minutes"}),
    ("fact_4", {"text": "Eating protein at breakfast helps maintain steady energy levels throughout the day"}),
    ("fact_5", {"text": "Blue light from screens can disrupt your circadian rhythm and sleep"}),
    ("fact_6", {"text": "Walking for 30 minutes daily can improve cardiovascular health"}),
    ("fact_7", {"text": "Magnesium-rich foods like nuts and leafy greens can help with muscle cramps"}),
    ("fact_8", {"text": "A consistent sleep schedule, even on weekends, improves overall sleep quality"}),
]

for key, value in wellness_facts:
    semantic_store.put(namespace, key, value)

print(f"Stored {len(wellness_facts)} wellness facts in semantic memory")
Stored 8 wellness facts in semantic memory
# Search semantically - notice we don't need exact matches!

queries = [
    "My head hurts, what should I do?",
    "How can I get better rest at night?",
    "I'm feeling stressed and anxious",
    "What should I eat in the morning?",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = semantic_store.search(namespace, query=query, limit=2)
    for r in results:
        print(f"   {r.value['text']} (score: {r.score:.3f})")
Query: My head hurts, what should I do?
   Drinking water can help relieve headaches caused by dehydration (score: 0.327)
   Magnesium-rich foods like nuts and leafy greens can help with muscle cramps (score: 0.173)

Query: How can I get better rest at night?
   Regular exercise improves sleep quality and helps you fall asleep faster (score: 0.463)
   A consistent sleep schedule, even on weekends, improves overall sleep quality (score: 0.426)

Query: I'm feeling stressed and anxious
   Deep breathing exercises can reduce stress and anxiety within minutes (score: 0.415)
   Drinking water can help relieve headaches caused by dehydration (score: 0.224)

Query: What should I eat in the morning?
   Eating protein at breakfast helps maintain steady energy levels throughout the day (score: 0.467)
   Walking for 30 minutes daily can improve cardiovascular health (score: 0.249)
Task 7: Building Semantic Wellness Knowledge Base
Let's load the HealthWellnessGuide.txt and create a semantic knowledge base that our agent can search.

This is similar to RAG from Session 4, but now using LangGraph's Store API instead of a separate vector database.

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and chunk the wellness document
loader = TextLoader("data/HealthWellnessGuide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded and split into {len(chunks)} chunks")
print(f"\nSample chunk:\n{chunks[0].page_content[:200]}...")
Loaded and split into 45 chunks

Sample chunk:
The Personal Wellness Guide
A Comprehensive Resource for Health and Well-being

PART 1: EXERCISE AND MOVEMENT

Chapter 1: Understanding Exercise Basics

Exercise is one of the most important things yo...
# Store chunks in semantic memory
knowledge_namespace = ("wellness", "knowledge")

for i, chunk in enumerate(chunks):
    semantic_store.put(
        knowledge_namespace,
        f"chunk_{i}",
        {"text": chunk.page_content, "source": "HealthWellnessGuide.txt"}
    )

print(f"Stored {len(chunks)} chunks in semantic knowledge base")
Stored 45 chunks in semantic knowledge base
# Build a semantic search wellness chatbot

class SemanticState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def semantic_wellness_chatbot(state: SemanticState, config: RunnableConfig, *, store: BaseStore):
    """A wellness chatbot that retrieves relevant facts using semantic search."""
    user_message = state["messages"][-1].content
    
    # Search for relevant knowledge
    knowledge_results = store.search(
        ("wellness", "knowledge"),
        query=user_message,
        limit=3
    )
    
    # Build context from retrieved knowledge
    if knowledge_results:
        knowledge_text = "\n\n".join([f"- {r.value['text']}" for r in knowledge_results])
        system_msg = f"""You are a Personal Wellness Assistant with access to a wellness knowledge base.

Relevant information from your knowledge base:
{knowledge_text}

Use this information to answer the user's question. If the information doesn't directly answer their question, use your general knowledge but mention what you found."""
    else:
        system_msg = "You are a Personal Wellness Assistant. Answer wellness questions helpfully."
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build and compile
builder3 = StateGraph(SemanticState)
builder3.add_node("chatbot", semantic_wellness_chatbot)
builder3.add_edge(START, "chatbot")
builder3.add_edge("chatbot", END)

semantic_graph = builder3.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Semantic wellness chatbot ready")
Semantic wellness chatbot ready
# Test semantic retrieval
config = {"configurable": {"thread_id": "semantic_thread_1"}}

questions = [
    "What exercises can help with lower back pain?",
    "How can I improve my sleep quality?",
    "What should I eat for better gut health?",
]

for q in questions:
    response = semantic_graph.invoke(
        {"messages": [HumanMessage(content=q)], "user_id": "test_user"},
        config
    )
    print(f"\nUser: {q}")
    print(f"Assistant: {response['messages'][-1].content[:500]}...")
User: What exercises can help with lower back pain?
Assistant: Here are some effective exercises that can help alleviate lower back pain:

1. **Cat-Cow Stretch**: 
   - Start on your hands and knees.
   - Alternate between arching your back up (like a cat) and letting it sag down (like a cow).
   - Perform 10-15 repetitions.

2. **Bird Dog**: 
   - From a hands and knees position, extend your opposite arm and leg while keeping your core engaged.
   - Hold for 5 seconds, then switch sides.
   - Aim for 10 repetitions per side.

3. **Partial Crunches**: 
   -...

User: How can I improve my sleep quality?
Assistant: Improving your sleep quality can be achieved through several essential sleep hygiene practices and creating an optimal sleep environment. Here are some effective strategies:

### Sleep Hygiene Practices:
1. **Maintain a Consistent Sleep Schedule**: Go to bed and wake up at the same time every day, even on weekends, to regulate your body's internal clock.
2. **Create a Relaxing Bedtime Routine**: Engage in calming activities before bed, such as reading, gentle stretching, or taking a warm bath, t...

User: What should I eat for better gut health?
Assistant: To support better gut health, focus on incorporating a variety of foods that promote a healthy digestive system. Here are some key food categories to include in your diet:

### Fermented Foods:
- **Yogurt**: Contains probiotics that can help balance gut bacteria.
- **Kimchi**: A spicy fermented vegetable dish rich in probiotics.
- **Kombucha**: A fermented tea that provides beneficial bacteria.
- **Miso**: A fermented soybean paste that can be used in soups and dressings.

### Prebiotic Foods:
-...
Task 8: Episodic Memory (Few-Shot Learning)
Episodic memory stores past experiences and uses them as examples for future tasks. In humans, this is like remembering the first time you successfully gave someone good advice and using that experience to guide future interactions.

For AI agents, episodic memory is often implemented as few-shot examples - showing the model examples of correct behavior so it can learn patterns.

The CoALA Framework Perspective
"Facts can be written to semantic memory, whereas experiences can be written to episodic memory."

Key insight: Sometimes it's easier to show than tell. LLMs learn well from examples!

# Create episodic memory with past successful interactions
episodic_namespace = ("agent", "episodes")

successful_episodes = [
    {
        "situation": "User asked about starting exercise with joint pain",
        "input": "I want to start exercising but my knees hurt. What should I do?",
        "output": """For knee-friendly exercise, I'd recommend starting with low-impact activities:

1. **Swimming or water aerobics** - Water supports your joints while providing great cardio
2. **Cycling** - Either stationary or regular bikes are gentle on knees
3. **Elliptical machine** - Provides cardio without impact

Start slowly with 15-20 minutes and gradually increase. If you experience pain, stop and consult a healthcare provider. Would you like specific exercises for any of these?""",
        "feedback": "User appreciated the specific alternatives and safety advice"
    },
    {
        "situation": "User struggling with sleep and stress",
        "input": "I can't fall asleep because I'm always thinking about work. Help!",
        "output": """Racing thoughts at bedtime is common, especially with work stress. Here's a proven approach:

**Before bed:**
- Write down tomorrow's to-do list (gets thoughts out of your head)
- Set a "worry time" earlier in the evening - not at bedtime

**At bedtime:**
- Try the 4-7-8 breathing technique: Inhale 4 counts, hold 7, exhale 8
- Use the "cognitive shuffle" - think of random, unrelated objects (apple, guitar, mountain...)

Which of these would you like to try first?""",
        "feedback": "User found the structured approach and specific techniques helpful"
    },
    {
        "situation": "User asking about healthy eating on a budget",
        "input": "How can I eat healthy when I don't have much money?",
        "output": """Eating healthy on a budget is absolutely possible! Here are practical strategies:

**Budget-friendly nutritious foods:**
- Eggs, beans, lentils (cheap protein)
- Frozen vegetables (same nutrition as fresh, longer lasting)
- Oats, rice, whole grain bread
- Canned fish (tuna, salmon)

**Money-saving tips:**
- Buy in bulk when possible
- Plan meals around sales
- Cook in batches and freeze portions

What's your typical weekly food budget? I can help you create a specific meal plan.""",
        "feedback": "User valued the practical, actionable advice without judgment"
    },
]

for i, episode in enumerate(successful_episodes):
    semantic_store.put(
        episodic_namespace,
        f"episode_{i}",
        {
            "text": episode["situation"],  # Used for semantic search
            **episode
        }
    )

print(f"Stored {len(successful_episodes)} episodic memories (past successful interactions)")
Stored 3 episodic memories (past successful interactions)
class EpisodicState(TypedDict):
    messages: Annotated[list, add_messages]


def episodic_wellness_chatbot(state: EpisodicState, config: RunnableConfig, *, store: BaseStore):
    """A chatbot that learns from past successful interactions."""
    user_question = state["messages"][-1].content
    
    # Search for similar past experiences
    similar_episodes = store.search(
        ("agent", "episodes"),
        query=user_question,
        limit=1
    )
    
    # Build few-shot examples from past episodes
    if similar_episodes:
        episode = similar_episodes[0].value
        few_shot_example = f"""Here's an example of a similar wellness question I handled well:

User asked: {episode['input']}

My response was:
{episode['output']}

The user feedback was: {episode['feedback']}

Use this as inspiration for the style, structure, and tone of your response, but tailor it to the current question."""
        
        system_msg = f"""You are a Personal Wellness Assistant. Learn from your past successes:

{few_shot_example}"""
    else:
        system_msg = "You are a Personal Wellness Assistant. Be helpful, specific, and supportive."
    
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build the episodic memory graph
builder4 = StateGraph(EpisodicState)
builder4.add_node("chatbot", episodic_wellness_chatbot)
builder4.add_edge(START, "chatbot")
builder4.add_edge("chatbot", END)

episodic_graph = builder4.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Episodic memory chatbot ready")
Episodic memory chatbot ready
# Test episodic memory - similar question to stored episode
config = {"configurable": {"thread_id": "episodic_thread_1"}}

response = episodic_graph.invoke(
    {"messages": [HumanMessage(content="I want to exercise more but I have a bad hip. What can I do?")]},
    config
)

print("User: I want to exercise more but I have a bad hip. What can I do?")
print(f"\nAssistant: {response['messages'][-1].content}")
print("\nNotice: The response structure mirrors the successful knee pain episode!")
User: I want to exercise more but I have a bad hip. What can I do?

Assistant: It's great that you want to exercise more! For a bad hip, focusing on low-impact activities can help you stay active while minimizing discomfort. Here are some options to consider:

1. **Swimming or water aerobics** - The buoyancy of water reduces stress on your hip joints while providing a full-body workout.
2. **Cycling** - Using a stationary bike or cycling outdoors can be gentle on your hips and still give you a good cardiovascular workout.
3. **Walking** - Start with short, flat walks and gradually increase your distance as you feel comfortable. Consider using supportive shoes.
4. **Yoga or Pilates** - These practices can improve flexibility and strength without putting too much strain on your hips. Look for classes that focus on gentle movements.
5. **Resistance training** - Using resistance bands or light weights can help strengthen the muscles around your hip, providing better support.

Start with 15-20 minutes of activity and listen to your body. If you feel any pain, it's important to stop and consult a healthcare provider. Would you like more specific exercises or tips for any of these activities?

Notice: The response structure mirrors the successful knee pain episode!
Task 9: Procedural Memory (Self-Improving Agent)
Procedural memory stores the rules and instructions that guide behavior. In humans, this is like knowing how to give good advice - it's internalized knowledge about performing tasks.

For AI agents, procedural memory often means self-modifying prompts. The agent can:

Store its current instructions in the memory store
Reflect on feedback from interactions
Update its own instructions to improve
The Reflection Pattern
User feedback: "Your advice is too long and complicated"
         ↓
Agent reflects on current instructions
         ↓
Agent updates instructions: "Keep advice concise and actionable"
         ↓
Future responses use updated instructions
# Initialize procedural memory with base instructions
procedural_namespace = ("agent", "instructions")

initial_instructions = """You are a Personal Wellness Assistant.

Guidelines:
- Be supportive and non-judgmental
- Provide evidence-based wellness information
- Ask clarifying questions when needed
- Encourage healthy habits without being preachy"""

semantic_store.put(
    procedural_namespace,
    "wellness_assistant",
    {"instructions": initial_instructions, "version": 1}
)

print("Initialized procedural memory with base instructions")
print(f"\nCurrent Instructions (v1):\n{initial_instructions}")
Initialized procedural memory with base instructions

Current Instructions (v1):
You are a Personal Wellness Assistant.

Guidelines:
- Be supportive and non-judgmental
- Provide evidence-based wellness information
- Ask clarifying questions when needed
- Encourage healthy habits without being preachy
class ProceduralState(TypedDict):
    messages: Annotated[list, add_messages]
    feedback: str  # Optional feedback from user


def get_instructions(store: BaseStore) -> tuple[str, int]:
    """Retrieve current instructions from procedural memory."""
    item = store.get(("agent", "instructions"), "wellness_assistant")
    if item is None:
        return "You are a helpful wellness assistant.", 0
    return item.value["instructions"], item.value["version"]


def procedural_assistant_node(state: ProceduralState, config: RunnableConfig, *, store: BaseStore):
    """Respond using current procedural instructions."""
    instructions, version = get_instructions(store)
    
    messages = [SystemMessage(content=instructions)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def reflection_node(state: ProceduralState, config: RunnableConfig, *, store: BaseStore):
    """Reflect on feedback and update instructions if needed."""
    feedback = state.get("feedback", "")
    
    if not feedback:
        return {}  # No feedback, no update needed
    
    # Get current instructions
    current_instructions, version = get_instructions(store)
    
    # Ask the LLM to reflect and improve instructions
    reflection_prompt = f"""You are improving a wellness assistant's instructions based on user feedback.

Current Instructions:
{current_instructions}

User Feedback:
{feedback}

Based on this feedback, provide improved instructions. Keep the same general format but incorporate the feedback.
Only output the new instructions, nothing else."""
    
    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    new_instructions = response.content
    
    # Update procedural memory with new instructions
    store.put(
        ("agent", "instructions"),
        "wellness_assistant",
        {"instructions": new_instructions, "version": version + 1}
    )
    
    print(f"\nInstructions updated to version {version + 1}")
    return {}


def should_reflect(state: ProceduralState) -> str:
    """Decide whether to reflect on feedback."""
    if state.get("feedback"):
        return "reflect"
    return "end"


# Build the procedural memory graph
builder5 = StateGraph(ProceduralState)
builder5.add_node("assistant", procedural_assistant_node)
builder5.add_node("reflect", reflection_node)

builder5.add_edge(START, "assistant")
builder5.add_conditional_edges("assistant", should_reflect, {"reflect": "reflect", "end": END})
builder5.add_edge("reflect", END)

procedural_graph = builder5.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Procedural memory graph ready (with self-improvement capability)")
Procedural memory graph ready (with self-improvement capability)
# Test with initial instructions
config = {"configurable": {"thread_id": "procedural_thread_1"}}

response = procedural_graph.invoke(
    {
        "messages": [HumanMessage(content="How can I reduce stress?")],
        "feedback": ""  # No feedback yet
    },
    config
)

print("User: How can I reduce stress?")
print(f"\nAssistant (v1 instructions):\n{response['messages'][-1].content}")
User: How can I reduce stress?

Assistant (v1 instructions):
Reducing stress is a great goal, and there are several effective strategies you can try. Here are some evidence-based methods:

1. **Mindfulness and Meditation**: Practicing mindfulness or meditation can help you stay present and reduce anxiety. Even a few minutes a day can make a difference.

2. **Physical Activity**: Regular exercise is a powerful stress reliever. It can boost your mood and improve your overall well-being. What type of physical activity do you enjoy?

3. **Deep Breathing Exercises**: Simple deep breathing techniques can help calm your mind and body. Try inhaling deeply for a count of four, holding for four, and exhaling for four.

4. **Connect with Others**: Talking to friends or family can provide support and help you feel less isolated. Do you have someone you feel comfortable reaching out to?

5. **Time Management**: Sometimes, stress comes from feeling overwhelmed. Organizing your tasks and setting priorities can help you feel more in control. Have you tried any time management techniques before?

6. **Healthy Eating**: A balanced diet can impact your mood and energy levels. Incorporating more fruits, vegetables, and whole grains can be beneficial. How do you feel about your current eating habits?

7. **Sleep Hygiene**: Ensuring you get enough quality sleep is crucial for stress management. Establishing a calming bedtime routine can help. How is your sleep quality lately?

8. **Limit Caffeine and Alcohol**: Both can increase feelings of anxiety and stress. Being mindful of your intake can help. Do you consume these regularly?

Feel free to share what you’ve tried or what resonates with you, and we can explore more tailored strategies!
# Now provide feedback - the agent will update its own instructions!
response = procedural_graph.invoke(
    {
        "messages": [HumanMessage(content="How can I reduce stress?")],
        "feedback": "Your responses are too long. Please be more concise and give me 3 actionable tips maximum."
    },
    {"configurable": {"thread_id": "procedural_thread_2"}}
)
Instructions updated to version 2
# Check the updated instructions
new_instructions, version = get_instructions(semantic_store)
print(f"Updated Instructions (v{version}):\n")
print(new_instructions)
Updated Instructions (v2):

You are a Personal Wellness Assistant.

Guidelines:
- Be supportive and non-judgmental
- Provide evidence-based wellness information
- Ask clarifying questions when needed
- Encourage healthy habits without being preachy
- Keep responses concise, offering a maximum of 3 actionable tips.
# Test with updated instructions - should be more concise now!
response = procedural_graph.invoke(
    {
        "messages": [HumanMessage(content="How can I sleep better?")],
        "feedback": ""  # No feedback this time
    },
    {"configurable": {"thread_id": "procedural_thread_3"}}
)

print(f"User: How can I sleep better?")
print(f"\nAssistant (v{version} instructions - after feedback):")
print(response['messages'][-1].content)
print("\nNotice: The response should now be more concise based on the feedback!")
User: How can I sleep better?

Assistant (v2 instructions - after feedback):
Improving your sleep can have a significant impact on your overall wellness. Here are three actionable tips to help you sleep better:

1. **Establish a Consistent Sleep Schedule**: Try to go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's internal clock.

2. **Create a Relaxing Bedtime Routine**: Engage in calming activities before bed, such as reading, gentle stretching, or meditation. Avoid screens for at least 30 minutes before sleep, as blue light can interfere with melatonin production.

3. **Optimize Your Sleep Environment**: Make your bedroom conducive to sleep by keeping it dark, cool, and quiet. Consider using blackout curtains, earplugs, or a white noise machine if needed.

Would you like more specific strategies or information on any of these tips?

Notice: The response should now be more concise based on the feedback!
Task 10: Unified Wellness Memory Agent
Now let's combine all 5 memory types into a unified wellness agent:

Short-term: Remembers current conversation (checkpointer)
Long-term: Stores user profile across sessions (store + namespace)
Semantic: Retrieves relevant wellness knowledge (store + embeddings)
Episodic: Uses past successful interactions as examples (store + search)
Procedural: Adapts behavior based on feedback (store + reflection)
Memory Retrieval Flow
User Query: "What exercises can help my back pain?"
              │
              ▼
┌─────────────────────────────────────────────────┐
│  1. PROCEDURAL: Get current instructions         │
│  2. LONG-TERM: Load user profile (conditions)    │
│  3. SEMANTIC: Search wellness knowledge          │
│  4. EPISODIC: Find similar past interactions     │
│  5. SHORT-TERM: Include conversation history     │
└─────────────────────────────────────────────────┘
              │
              ▼
        Generate personalized, informed response
class UnifiedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    feedback: str


def unified_wellness_assistant(state: UnifiedState, config: RunnableConfig, *, store: BaseStore):
    """An assistant that uses all five memory types."""
    user_id = state["user_id"]
    user_message = state["messages"][-1].content
    
    # 1. PROCEDURAL: Get current instructions
    instructions_item = store.get(("agent", "instructions"), "wellness_assistant")
    base_instructions = instructions_item.value["instructions"] if instructions_item else "You are a helpful wellness assistant."
    
    # 2. LONG-TERM: Get user profile
    profile_items = list(store.search((user_id, "profile")))
    pref_items = list(store.search((user_id, "preferences")))
    profile_text = "\n".join([f"- {p.key}: {p.value}" for p in profile_items]) if profile_items else "No profile stored."
    
    # 3. SEMANTIC: Search for relevant knowledge
    relevant_knowledge = store.search(("wellness", "knowledge"), query=user_message, limit=2)
    knowledge_text = "\n".join([f"- {r.value['text'][:200]}..." for r in relevant_knowledge]) if relevant_knowledge else "No specific knowledge found."
    
    # 4. EPISODIC: Find similar past interactions
    similar_episodes = store.search(("agent", "episodes"), query=user_message, limit=1)
    if similar_episodes:
        ep = similar_episodes[0].value
        episode_text = f"Similar past interaction:\nUser: {ep.get('input', 'N/A')}\nResponse style: {ep.get('feedback', 'N/A')}"
    else:
        episode_text = "No similar past interactions found."
    
    # Build comprehensive system message
    system_message = f"""{base_instructions}

=== USER PROFILE ===
{profile_text}

=== RELEVANT WELLNESS KNOWLEDGE ===
{knowledge_text}

=== LEARNING FROM EXPERIENCE ===
{episode_text}

Use all of this context to provide the best possible personalized response."""
    
    # 5. SHORT-TERM: Full conversation history is automatically managed by the checkpointer
    # Use summarization for long conversations
    trimmed_messages = summarize_conversation(state["messages"], max_messages=6)
    
    messages = [SystemMessage(content=system_message)] + trimmed_messages
    response = llm.invoke(messages)
    return {"messages": [response]}


def unified_feedback_node(state: UnifiedState, config: RunnableConfig, *, store: BaseStore):
    """Update procedural memory based on feedback."""
    feedback = state.get("feedback", "")
    if not feedback:
        return {}
    
    item = store.get(("agent", "instructions"), "wellness_assistant")
    if item is None:
        return {}
    
    current = item.value
    reflection_prompt = f"""Update these instructions based on feedback:

Current: {current['instructions']}
Feedback: {feedback}

Output only the updated instructions."""
    
    response = llm.invoke([HumanMessage(content=reflection_prompt)])
    store.put(
        ("agent", "instructions"),
        "wellness_assistant",
        {"instructions": response.content, "version": current["version"] + 1}
    )
    print(f"Procedural memory updated to v{current['version'] + 1}")
    return {}


def unified_route(state: UnifiedState) -> str:
    return "feedback" if state.get("feedback") else "end"


# Build the unified graph
unified_builder = StateGraph(UnifiedState)
unified_builder.add_node("assistant", unified_wellness_assistant)
unified_builder.add_node("feedback", unified_feedback_node)

unified_builder.add_edge(START, "assistant")
unified_builder.add_conditional_edges("assistant", unified_route, {"feedback": "feedback", "end": END})
unified_builder.add_edge("feedback", END)

# Compile with both checkpointer (short-term) and store (all other memory types)
unified_graph = unified_builder.compile(
    checkpointer=MemorySaver(),
    store=semantic_store
)

print("Unified wellness assistant ready with all 5 memory types!")
Unified wellness assistant ready with all 5 memory types!
# Test the unified assistant
config = {"configurable": {"thread_id": "unified_thread_1"}}

# First interaction - should use semantic + long-term + episodic memory
response = unified_graph.invoke(
    {
        "messages": [HumanMessage(content="What exercises would you recommend for my back?")],
        "user_id": "user_sarah",  # Sarah has a bad knee in her profile!
        "feedback": ""
    },
    config
)

print("User: What exercises would you recommend for my back?")
print(f"\nAssistant: {response['messages'][-1].content}")
print("\n" + "="*60)
print("Memory types used:")
print("  Long-term: Knows Sarah has a bad knee")
print("  Semantic: Retrieved back exercise info from knowledge base")
print("  Episodic: May use similar joint pain episode as reference")
print("  Procedural: Following current instructions")
print("  Short-term: Will remember this in follow-up questions")
User: What exercises would you recommend for my back?

Assistant: It's great that you're looking to support your back health! Here are three gentle exercises that can help relieve lower back pain:

1. **Cat-Cow Stretch**: Start on your hands and knees. Alternate between arching your back up (like a cat) and letting it sag down (like a cow). Repeat this for 10-15 repetitions to help increase flexibility and relieve tension.

2. **Bird-Dog**: From the same hands-and-knees position, extend one arm forward and the opposite leg back, keeping your back straight. Hold for a few seconds, then switch sides. Aim for 8-10 repetitions on each side to strengthen your core and lower back.

3. **Child’s Pose**: Kneel on the floor, sit back on your heels, and stretch your arms forward on the ground. Hold this position for 20-30 seconds to gently stretch your back and promote relaxation.

Always listen to your body and stop if you feel any pain. Have you tried any of these exercises before?

============================================================
Memory types used:
  Long-term: Knows Sarah has a bad knee
  Semantic: Retrieved back exercise info from knowledge base
  Episodic: May use similar joint pain episode as reference
  Procedural: Following current instructions
  Short-term: Will remember this in follow-up questions
# Follow-up question (tests short-term memory)
response = unified_graph.invoke(
    {
        "messages": [HumanMessage(content="Can you show me how to do the first one?")],
        "user_id": "user_sarah",
        "feedback": ""
    },
    config  # Same thread
)

print("User: Can you show me how to do the first one?")
print(f"\nAssistant: {response['messages'][-1].content}")
print("\nNotice: The agent remembers the context from the previous message!")
User: Can you show me how to do the first one?

Assistant: Absolutely! Here’s a step-by-step guide for the **Cat-Cow Stretch**:

1. **Start Position**: Begin on your hands and knees on a comfortable surface, like a yoga mat. Your wrists should be directly under your shoulders, and your knees should be under your hips.

2. **Cat Position**: Inhale deeply. As you exhale, round your back towards the ceiling, tucking your chin to your chest. This is the "cat" position. Hold for a moment.

3. **Cow Position**: Inhale again. As you exhale, arch your back, allowing your belly to drop towards the floor, and lift your head and tailbone towards the ceiling. This is the "cow" position. Hold for a moment.

4. **Repeat**: Continue to alternate between these two positions for 10-15 repetitions, synchronizing your breath with your movements.

This stretch is great for increasing flexibility and relieving tension in your back. How does that sound? Would you like tips on how to incorporate it into your routine?

Notice: The agent remembers the context from the previous message!
❓ Question #3:
How would you decide what constitutes a "successful" wellness interaction worth storing as an episode? What metadata should you store alongside the episode?

Consider:

Explicit feedback (thumbs up) vs implicit signals
User engagement (did they ask follow-up questions?)
Objective outcomes vs subjective satisfaction
Privacy implications of storing interaction data
Answer:
A “successful” wellness interaction is one where the agent’s response demonstrably helped the user move forward, not just where the answer sounded good. In practice, I would decide success using a combination of explicit signals, implicit behavioral signals, and contextual outcomes, rather than relying on any single indicator.

The strongest signal is explicit feedback, such as a thumbs-up, a positive rating, or a direct statement like “this was helpful” or “that worked.” These signals are unambiguous and should almost always qualify an interaction for episodic storage, especially when the advice involves safety, behavior change, or emotional support. However, explicit feedback is relatively rare, so it cannot be the only criterion.

That is where implicit signals matter. Continued engagement is a strong proxy for success in wellness contexts. If the user asks follow-up questions, requests clarification, or tries to apply the advice (“I tried the breathing exercise and…”) it suggests the guidance was actionable and relevant. Conversely, abrupt topic switches or conversation drop-off may indicate low value. Reuse of advice patterns across sessions (e.g., returning with similar goals) is another implicit signal that the prior interaction was effective.

Objective outcomes are harder to measure in wellness systems but are still important when available. For example, improvements in tracked metrics (better sleep scores, reduced stress ratings, increased activity consistency) can retrospectively validate that an interaction was successful. These signals are usually delayed, so they are best used as reinforcement for episodic promotion rather than real-time gating.

When storing an episode, the goal is not to save the entire conversation, but to capture the experience pattern that worked. The metadata I would store alongside an episode includes:

The user intent or situation (e.g., “starting exercise with joint pain”)
The user input (sanitized and minimized)
The assistant response (or a summarized version)
The outcome signal (explicit feedback, follow-up engagement, metric improvement)
The contextual constraints that mattered (injury, allergy, stress level)
A confidence score or success label derived from the signals
A timestamp and optional expiration/TTL
A privacy level (e.g., low-risk behavioral vs sensitive health context)
Privacy considerations are critical for episodic memory because it stores experiences, not just facts. Episodes should be minimized, anonymized where possible, and purpose-bound. Highly sensitive or emotionally vulnerable interactions should either not be stored at all, or stored only as abstracted patterns (e.g., “structured, empathetic response with grounding techniques worked”) without verbatim user content. Ideally, episodic storage should be gated by user consent or limited to agent-centric learnings (style, structure, tone) rather than personal details.

In summary, an interaction is worth storing as an episode when it shows clear evidence of usefulness, can teach the agent something reusable, and can be stored in a privacy-aware, abstracted form that improves future responses without over-retaining personal data.

❓ Question #4:
For a production wellness assistant, which memory types need persistent storage (PostgreSQL) vs in-memory? How would you handle memory across multiple agent instances (e.g., Exercise Agent, Nutrition Agent, Sleep Agent)?

Consider:

Which memories are user-specific vs shared?
Consistency requirements across agents
Memory expiration and cleanup policies
Namespace strategy for multi-agent systems
Answer:
For production, anything that must survive restarts and be visible to all agent instances should be persistent (PostgreSQL), and only truly ephemeral/cached data should be in-memory.

Persistent (PostgreSQL): i. Short-term memory (checkpoints): use PostgresSaver so a thread can resume and any instance can continue the conversation. ii. Long-term memory: user profile + preferences (allergies, conditions, injuries, goals). This is user-specific and must be consistent everywhere. iii. Episodic memory: “what worked before” episodes per user (and often per domain). Persist so lessons carry across sessions and devices. iv. Procedural memory: the current instruction set / behavior rules (global + per-agent variants). Persist with versioning so every agent follows the same updated guidance. v. Semantic memory: the shared wellness knowledge base and embeddings index should be persistent; user-specific semantic notes (if you store any) should also be persistent.

In-memory (only): Local dev stores, short-lived caches of retrieval results, and transient intermediate state you can recompute. Not user memory.

Handling memory across multiple agent instances (Exercise/Nutrition/Sleep): i. Use one shared Postgres-backed store + one shared checkpointer backend for all agents. ii. Separate “shared” vs “user” data with namespaces: -Shared KB: ("kb", "wellness") - User core: (user_id, "profile"), (user_id, "preferences") - Domain memories: (user_id, "exercise", "episodes"), (user_id, "nutrition", "episodes"), (user_id, "sleep", "episodes") - Procedural instructions: ("agent", "instructions", "<agent_name>") (and optionally ("agent", "policies") for global guardrails) iii. Consistency: treat profile/preferences as the single source of truth, use versioning/optimistic concurrency for updates, and restrict agents to writing only in their domain namespaces. iv. Expiration/cleanup: TTL for checkpoints and episodic memories; keep top-K episodes per domain; user-controlled delete/export for privacy.

🏗️ Activity #2: Wellness Memory Dashboard
Build a wellness tracking system that:

Tracks wellness metrics over time (mood, energy, sleep quality)
Uses semantic memory to find relevant advice
Uses episodic memory to recall what worked before
Uses procedural memory to adapt advice style
Provides a synthesized "wellness summary"
Requirements:
Store at least 3 wellness metrics per user
Track metrics over multiple "days" (simulated)
Agent should reference historical data in responses
Generate a personalized wellness summary
Step 1: Define wellness metrics schema and storage functions

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage


# Step 1: Define wellness metrics schema and storage functions

def _metrics_ns(user_id: str) -> tuple:
    return (user_id, "metrics")

def _metric_key(date: str, metric_type: str) -> str:
    # Example key: "2026-02-01:sleep_quality"
    return f"{date}:{metric_type}"

def log_wellness_metric(store: BaseStore, user_id: str, date: str, metric_type: str, value: float, notes: str = ""):
    """Log a wellness metric for a user."""
    if not user_id:
        raise ValueError("user_id is required")
    if not date:
        raise ValueError("date is required (YYYY-MM-DD)")
    if not metric_type:
        raise ValueError("metric_type is required")
    if value is None:
        raise ValueError("value is required")

    ns = _metrics_ns(user_id)
    key = _metric_key(date, metric_type)

    store.put(
        ns,
        key,
        {
            "date": date,
            "metric_type": metric_type,
            "value": float(value),
            "notes": notes or ""
        }
    )

def get_wellness_history(store: BaseStore, user_id: str, metric_type: str = None, days: int = 7) -> list:
    """Get wellness history for a user. Returns most recent entries (simple filter + sort)."""
    ns = _metrics_ns(user_id)
    items = list(store.search(ns))  # InMemoryStore returns all items; we filter manually

    records = []
    for it in items:
        v = it.value
        if metric_type and v.get("metric_type") != metric_type:
            continue
        records.append(v)

    # Sort by date ascending (string sort works for YYYY-MM-DD)
    records.sort(key=lambda r: (r.get("date", ""), r.get("metric_type", "")))

    # Keep only last N days worth (simple: keep last N records per metric_type is more complex; OK for activity)
    # We'll just keep last ~days*3 records assuming 3 metrics/day
    if days is not None:
        records = records[-(days * 3):]

    return records
Step 2: Create sample wellness data for a user (simulate a week)

# Step 2: Create sample wellness data for a user (simulate a week)

# Use the same store you used for semantic/episodic/procedural memory in earlier tasks.
# Here I assume it's called `semantic_store`.
# If your variable is different, rename below.
store = semantic_store

user_id = "user_sarah"

sample_week = [
    # date, mood(1-10), energy(1-10), sleep_quality(1-10), notes
    ("2026-02-01", 6, 5, 6, "Busy day, late screen time"),
    ("2026-02-02", 5, 4, 5, "Woke up twice, work stress"),
    ("2026-02-03", 6, 5, 7, "Tried evening walk"),
    ("2026-02-04", 7, 6, 7, "Did 4-7-8 breathing before bed"),
    ("2026-02-05", 7, 6, 8, "No caffeine after 2pm"),
    ("2026-02-06", 6, 5, 6, "Ate late dinner"),
    ("2026-02-07", 8, 7, 8, "Consistent bedtime + short yoga"),
]

for date, mood, energy, sleep_q, notes in sample_week:
    log_wellness_metric(store, user_id, date, "mood", mood, notes)
    log_wellness_metric(store, user_id, date, "energy", energy, notes)
    log_wellness_metric(store, user_id, date, "sleep_quality", sleep_q, notes)

print("Sample week metrics stored for", user_id)
Sample week metrics stored for user_sarah
Step 3: Build a wellness dashboard agent that:

Retrieves user's wellness history
Searches for relevant advice based on patterns
Uses episodic memory for what worked before
Generates a personalized summary
# Step 3: Build a wellness dashboard agent
#   - Retrieves user's wellness history
#   - Searches for relevant advice based on patterns (semantic KB)
#   - Uses episodic memory for what worked before
#   - Uses procedural memory to adapt advice style
#   - Generates a personalized summary

class DashboardState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

def _trend_summary(records: list) -> str:
    """Lightweight trend extraction (no heavy stats)."""
    # Group by metric_type
    by_type = {}
    for r in records:
        by_type.setdefault(r["metric_type"], []).append(r)

    def avg(vals):
        return sum(vals) / max(len(vals), 1)

    lines = []
    for metric, rs in by_type.items():
        vals = [x["value"] for x in rs]
        start = vals[0] if vals else None
        end = vals[-1] if vals else None
        a = avg(vals)
        direction = ""
        if start is not None and end is not None:
            if end > start: direction = "improving"
            elif end < start: direction = "declining"
            else: direction = "stable"
        lines.append(f"- {metric}: avg={a:.1f}, start={start}, end={end} ({direction})")
    return "\n".join(lines) if lines else "No metrics found."

def _recent_notes(records: list, limit: int = 5) -> str:
    # Keep distinct day notes (they repeat across 3 metrics/day in our sample)
    seen = set()
    notes = []
    for r in sorted(records, key=lambda x: x["date"], reverse=True):
        key = (r["date"], r.get("notes", ""))
        if key in seen:
            continue
        seen.add(key)
        n = r.get("notes", "")
        if n:
            notes.append(f"- {r['date']}: {n}")
        if len(notes) >= limit:
            break
    return "\n".join(notes) if notes else "No notes."

def wellness_dashboard_node(state: DashboardState, config: RunnableConfig, *, store: BaseStore):
    user_id = state["user_id"]
    user_query = state["messages"][-1].content

    # 1) PROCEDURAL memory: get current instructions
    instr_item = store.get(("agent", "instructions"), "wellness_assistant")
    instructions = instr_item.value["instructions"] if instr_item else "You are a helpful wellness assistant."

    # 2) Retrieve metrics history (last 7 days)
    history = get_wellness_history(store, user_id, days=7)
    trends = _trend_summary(history)
    notes = _recent_notes(history, limit=5)

    # Build a "pattern query" for semantic search (based on low areas)
    # Simple heuristic: if averages are low, ask KB for sleep/stress/energy advice
    pattern_query = f"Wellness patterns: {trends}. User asked: {user_query}"

    # 3) SEMANTIC memory: retrieve relevant advice from knowledge base
    kb_hits = store.search(("wellness", "knowledge"), query=pattern_query, limit=2)
    kb_text = "\n\n".join([f"- {h.value['text'][:220]}..." for h in kb_hits]) if kb_hits else "No KB hits."

    # 4) EPISODIC memory: retrieve a similar past successful episode (style pattern)
    ep_hits = store.search(("agent", "episodes"), query=user_query, limit=1)
    if ep_hits:
        ep = ep_hits[0].value
        episode_block = (
            f"Similar successful episode:\n"
            f"Situation: {ep.get('situation','')}\n"
            f"User asked: {ep.get('input','')}\n"
            f"What worked: {ep.get('feedback','')}"
        )
    else:
        episode_block = "No similar episode found."

    system_msg = f"""{instructions}

You are generating a Wellness Memory Dashboard summary.
Use the data below. Be specific about trends and reference the user's history.

=== METRICS (last 7 days) ===
{trends}

=== RECENT CONTEXT NOTES ===
{notes}

=== RELEVANT WELLNESS KNOWLEDGE (semantic retrieval) ===
{kb_text}

=== EPISODIC GUIDANCE (what worked before) ===
{episode_block}

Now respond to the user's request. Include:
1) A short summary of how mood/energy/sleep changed,
2) 2-3 likely contributing factors from notes,
3) 3 actionable recommendations tailored to the patterns.
"""

    messages = [SystemMessage(content=system_msg)] + state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


dashboard_builder = StateGraph(DashboardState)
dashboard_builder.add_node("dashboard", wellness_dashboard_node)
dashboard_builder.add_edge(START, "dashboard")
dashboard_builder.add_edge("dashboard", END)

dashboard_graph = dashboard_builder.compile(
    checkpointer=MemorySaver(),
    store=store
)

print("Wellness dashboard agent compiled.")
Wellness dashboard agent compiled.
Step 4: Test the dashboard Example: "Give me a summary of my wellness this week" Example: "I've been feeling tired lately. What might help?"

# Step 4: Test the dashboard

cfg = {"configurable": {"thread_id": "dashboard_thread_1"}}

# Example 1: weekly summary
resp = dashboard_graph.invoke(
    {"messages": [HumanMessage(content="Give me a summary of my wellness this week.")], "user_id": user_id},
    cfg
)
print("\n--- Dashboard: Weekly summary ---")
print(resp["messages"][-1].content)

# Example 2: tired lately
resp2 = dashboard_graph.invoke(
    {"messages": [HumanMessage(content="I've been feeling tired lately. What might help?")], "user_id": user_id},
    {"configurable": {"thread_id": "dashboard_thread_2"}}
)
print("\n--- Dashboard: Tired lately ---")
print(resp2["messages"][-1].content)
--- Dashboard: Weekly summary ---
Here's a summary of your wellness over the past week:

1) **Trends**: Your mood has shown improvement, increasing from an average of 6.0 to 7.0 by the end of the week. Sleep quality has also improved, moving from an average of 6.0 to 7.0. Your energy levels have remained stable at an average of 4.7.

2) **Contributing Factors**: 
   - You practiced 4-7-8 breathing before bed, which can help reduce anxiety and improve sleep quality.
   - You tried an evening walk, which may have positively influenced your mood and sleep.
   - Work stress impacted your sleep earlier in the week, but it seems you’ve found some strategies to manage it.

3) **Actionable Recommendations**:
   - Continue incorporating breathing exercises before bed to further enhance your sleep quality.
   - Aim for at least 30 minutes of movement daily, like your evening walks, to boost both mood and energy.
   - Consider setting a screen time limit in the evening to help improve your sleep consistency and quality.

Keep up the great work, and remember that small changes can lead to significant improvements!

--- Dashboard: Tired lately ---
Over the last week, your energy levels have remained stable at an average of 4.7, while your mood and sleep quality have shown improvement, moving from 6.0 to 7.0. This suggests that while your energy is consistent, there may still be room for enhancement.

Two likely contributing factors to your tiredness could be:
1. Work stress, which affected your sleep on February 2nd.
2. Late screen time on February 1st, which can disrupt sleep quality.

Here are three actionable recommendations to help boost your energy levels:

1. **Incorporate Quick Energy Boosters**: Try doing 10 jumping jacks or taking a brief walk around the block when you feel tired. These activities can help increase your energy levels quickly.

2. **Enhance Sleep Hygiene**: Maintain a consistent sleep schedule, even on weekends, and create a calming bedtime routine. Since you found 4-7-8 breathing helpful, consider incorporating it regularly before bed.

3. **Stay Hydrated**: Drink a glass of cold water when you wake up and throughout the day. Dehydration can lead to fatigue, so keeping hydrated can help improve your energy levels.

Feel free to let me know if you’d like more personalized strategies or if there’s anything specific you’d like to focus on!
Advanced Build : Router Agent (GPT-4o, structured output)
Goal: Route user queries → exercise | nutrition | sleep based on intent + user profile context.

from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

router_llm = ChatOpenAI(model="gpt-4o", temperature=0)

class RouterDecision(BaseModel):
    agent: Literal["exercise", "nutrition", "sleep"]
    reason: str

def router_agent(user_message: str, user_profile: str) -> RouterDecision:
    system = SystemMessage(
        content=f"""
You are a routing agent for a wellness system.

User profile:
{user_profile}

Decide which specialist agent should handle the query:
- exercise → workouts, injuries, movement
- nutrition → diet, meals, food
- sleep → sleep quality, insomnia, rest

Return a JSON decision.
"""
    )
    response = router_llm.with_structured_output(RouterDecision).invoke(
        [system, HumanMessage(content=user_message)]
    )
    return response
Advanced Build : Specialist Agents (GPT-4o-mini)
Each agent: Has domain-specific instructions, Reads shared memory, Writes to its own episodic + procedural namespaces

#Shared Model
from langchain_openai import ChatOpenAI
specialist_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Exercise Agent
def exercise_agent(state, store):
    instructions = store.get(
        ("exercise_agent", "instructions"), "v1"
    ) or {"instructions": "You are an exercise specialist."}

    episodes = store.search(("exercise_agent", "episodes"), limit=1)

    system = SystemMessage(content=f"""
{instructions['instructions']}

Recent successful episode:
{episodes[0].value['feedback'] if episodes else 'None'}

Use user profile and knowledge base.
""")

    response = specialist_llm.invoke([system] + state["messages"])
    return response

# Nutrition Agent
def nutrition_agent(state, store):
    instructions = store.get(
        ("nutrition_agent", "instructions"), "v1"
    ) or {"instructions": "You are a nutrition specialist."}

    episodes = store.search(("nutrition_agent", "episodes"), limit=1)

    system = SystemMessage(content=f"""
{instructions['instructions']}

Recent successful episode:
{episodes[0].value['feedback'] if episodes else 'None'}
""")

    response = specialist_llm.invoke([system] + state["messages"])
    return response


# Sleep Agent
def sleep_agent(state, store):
    instructions = store.get(
        ("sleep_agent", "instructions"), "v1"
    ) or {"instructions": "You are a sleep specialist."}

    episodes = store.search(("sleep_agent", "episodes"), limit=1)

    system = SystemMessage(content=f"""
{instructions['instructions']}

Recent successful episode:
{episodes[0].value['feedback'] if episodes else 'None'}
""")

    response = specialist_llm.invoke([system] + state["messages"])
    return response
Advanced Build : Shared vs Per-Agent Namespace Strategy
# Shared
(user_id, "profile")
("wellness", "knowledge")

# Per-agent
("exercise_agent", "instructions")
("exercise_agent", "episodes")

("nutrition_agent", "instructions")
("nutrition_agent", "episodes")

("sleep_agent", "instructions")
("sleep_agent", "episodes")
Advanced Build : Cross-Agent Episodic Memory Sharing
Each agent can read episodes from other agents when relevant.
Example: Nutrition agent learning from Exercise agent episodes.
def cross_agent_episodes(store, query: str, limit=1):
    episodes = []
    for agent in ["exercise_agent", "nutrition_agent", "sleep_agent"]:
        episodes.extend(
            store.search((agent, "episodes"), query=query, limit=limit)
        )
    return episodes
shared_episodes = cross_agent_episodes(store, state["messages"][-1].content)
Advanced Build : Orchestrator (Router → Specialist)
def multi_agent_orchestrator(state, store):
    user_id = state["user_id"]
    user_msg = state["messages"][-1].content

    profile_items = store.search((user_id, "profile"))
    profile_text = "\n".join(f"{p.key}: {p.value}" for p in profile_items)

    decision = router_agent(user_msg, profile_text)

    if decision.agent == "exercise":
        return exercise_agent(state, store)
    elif decision.agent == "nutrition":
        return nutrition_agent(state, store)
    else:
        return sleep_agent(state, store)
Advanced Build : Memory Dashboard (Text-Based, Required Fields)
def memory_dashboard(store, user_id: str):
    profile = list(store.search((user_id, "profile")))

    instruction_versions = {
        agent: store.get((agent, "instructions"), "v1")
        for agent in ["exercise_agent", "nutrition_agent", "sleep_agent"]
    }

    episode_counts = {
        agent: len(store.search((agent, "episodes")))
        for agent in ["exercise_agent", "nutrition_agent", "sleep_agent"]
    }

    return {
        "profile": {p.key: p.value for p in profile},
        "instruction_versions": instruction_versions,
        "episode_counts": episode_counts,
        "search": lambda q: store.search(("wellness", "knowledge"), query=q, limit=3)
    }
Summary
In this session, we explored the 5 memory types from the CoALA framework:

Memory Type	LangGraph Component	Scope	Wellness Use Case
Short-term	MemorySaver + thread_id	Within thread	Current consultation
Long-term	InMemoryStore + namespaces	Across threads	User profile, goals
Semantic	Store + embeddings + search()	Across threads	Knowledge retrieval
Episodic	Store + few-shot examples	Across threads	Past successful interactions
Procedural	Store + self-reflection	Across threads	Self-improving instructions
Key Takeaways:
Memory transforms chatbots into assistants - Persistence enables personalization
Different memory types serve different purposes - Choose based on your use case
Context management is critical - Trim and summarize to stay within limits
Episodic memory enables learning - Show, don't just tell
Procedural memory enables adaptation - Agents can improve themselves
Production Considerations:
Use PostgresSaver instead of MemorySaver for persistent checkpoints
Use PostgresStore instead of InMemoryStore for persistent long-term memory
Consider TTL (Time-to-Live) policies for automatic memory cleanup
Implement proper access controls for user data
Further Reading:
LangGraph Memory Documentation
CoALA Paper - Cognitive Architectures for Language Agents
LangGraph Platform - Managed infrastructure for production