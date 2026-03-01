
Deep Agents: Building Complex Agents for Long-Horizon Tasks
In this notebook, we'll explore Deep Agents - a new approach to building AI agents that can handle complex, multi-step tasks over extended periods. We'll implement all four key elements of Deep Agents while building on our Personal Wellness Assistant use case.

Learning Objectives:

Understand the four key elements of Deep Agents: Planning, Context Management, Subagent Spawning, and Long-term Memory
Implement each element progressively using the deepagents package
Learn to use Skills for progressive capability disclosure
Use the deepagents-cli for interactive agent sessions
Table of Contents:
Breakout Room #1: Deep Agent Foundations

Task 1: Dependencies & Setup
Task 2: Understanding Deep Agents
Task 3: Planning with Todo Lists
Task 4: Context Management with File Systems
Task 5: Basic Deep Agent
Question #1 & Question #2
Activity #1: Build a Research Agent
Breakout Room #2: Advanced Features & Integration

Task 6: Subagent Spawning
Task 7: Long-term Memory Integration
Task 8: Skills - On-Demand Capabilities
Task 9: Using deepagents-cli
Task 10: Building a Complete Deep Agent System
Question #3 & Question #4
Activity #2: Build a Wellness Coach Agent
🤝 Breakout Room #1
Deep Agent Foundations
Task 1: Dependencies & Setup
Before we begin, make sure you have:

API Keys for:

Anthropic (default for Deep Agents) or OpenAI
LangSmith (optional, for tracing)
Tavily (optional, for web search)
Dependencies installed via uv sync

For the CLI (Task 9): uv pip install deepagents-cli

Environment Setup
You can either:

Create a .env file with your API keys (recommended):
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
Or enter them interactively when prompted
# Core imports
import os
import getpass
from uuid import uuid4
from typing import Annotated, TypedDict, Literal

import nest_asyncio
nest_asyncio.apply()  # Required for async operations in Jupyter

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def get_api_key(env_var: str, prompt: str) -> str:
    """Get API key from environment or prompt user."""
    value = os.environ.get(env_var, "")
    if not value:
        value = getpass.getpass(prompt)
        if value:
            os.environ[env_var] = value
    return value
# Set Anthropic API Key (default for Deep Agents)
anthropic_key = get_api_key("ANTHROPIC_API_KEY", "Anthropic API Key: ")
if anthropic_key:
    print("Anthropic API key set")
else:
    print("Warning: No Anthropic API key configured")
Anthropic API key set
# Optional: OpenAI for alternative models and subagents
openai_key = get_api_key("OPENAI_API_KEY", "OpenAI API Key (press Enter to skip): ")
if openai_key:
    print("OpenAI API key set")
else:
    print("OpenAI API key not configured (optional)")
OpenAI API key set
# Optional: LangSmith for tracing
langsmith_key = get_api_key("LANGCHAIN_API_KEY", "LangSmith API Key (press Enter to skip): ")

if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"AIE9 - Deep Agents - {uuid4().hex[0:8]}"
    print(f"LangSmith tracing enabled. Project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled")
LangSmith tracing enabled. Project: AIE9 - Deep Agents - 85f20851
# Verify deepagents installation
# import sys
# print(sys.executable)
# print(sys.version)

# import sys, subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "deepagents"])

from deepagents import create_deep_agent
print("deepagents package imported successfully!")

# Test with a simple agent
test_agent = create_deep_agent()
result = test_agent.invoke({
    "messages": [{"role": "user", "content": "Say 'Deep Agents ready!' in exactly those words."}]
})
print(result["messages"][-1].content)
deepagents package imported successfully!
Deep Agents ready!
Task 2: Understanding Deep Agents
Deep Agents represent a shift from simple tool-calling loops to sophisticated agents that can handle complex, long-horizon tasks. They address four key challenges:

The Four Key Elements
Element	Challenge Addressed	Implementation
Planning	"What should I do?"	Todo lists that persist task state
Context Management	"What do I know?"	File systems for storing/retrieving info
Subagent Spawning	"Who can help?"	Task tool for delegating to specialists
Long-term Memory	"What did I learn?"	LangGraph Store for cross-session memory
Deep Agents vs Traditional Agents
Traditional Agent Loop:
┌─────────────────────────────────────┐
│  User Query                         │
│       ↓                             │
│  Think → Act → Observe → Repeat     │
│       ↓                             │
│  Response                           │
└─────────────────────────────────────┘
Problems: Context bloat, no delegation,
          loses track of complex tasks

Deep Agent Architecture:
┌─────────────────────────────────────────────────────────┐
│                    Deep Agent                           │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   PLANNING   │  │   CONTEXT    │  │   MEMORY     │   │
│  │              │  │  MANAGEMENT  │  │              │   │
│  │ write_todos  │  │              │  │   Store      │   │
│  │ update_todo  │  │  read_file   │  │  namespace   │   │
│  │ list_todos   │  │  write_file  │  │  get/put     │   │
│  │              │  │  edit_file   │  │              │   │
│  └──────────────┘  │  ls          │  └──────────────┘   │
│                    └──────────────┘                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SUBAGENT SPAWNING                   │   │
│  │                                                  │   │
│  │  task(prompt, tools, model, system_prompt)       │   │
│  │       ↓              ↓              ↓            │   │
│  │  ┌────────┐    ┌────────┐    ┌────────┐          │   │
│  │  │Research│    │Writing │    │Analysis│          │   │
│  │  │Subagent│    │Subagent│    │Subagent│          │   │
│  │  └────────┘    └────────┘    └────────┘          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
When to Use Deep Agents
Use Case	Traditional Agent	Deep Agent
Simple Q&A	✅	Overkill
Single-step tool use	✅	Overkill
Multi-step research	⚠️ May lose track	✅
Complex projects	❌ Context overflow	✅
Parallel task execution	❌	✅
Long-running sessions	❌	✅
Key Insight: "Planning is Context Engineering"
Deep Agents treat planning not as a separate phase, but as context engineering:

Todo lists aren't just task trackers—they're persistent context about what to do
File systems aren't just storage—they're extended memory beyond the context window
Subagents aren't just helpers—they're context isolation to prevent bloat
Task 3: Planning with Todo Lists
The first key element of Deep Agents is Planning. Instead of trying to hold all task state in the conversation, Deep Agents use structured todo lists.

Why Todo Lists?
Persistence: Tasks survive across conversation turns
Visibility: Both agent and user can see progress
Structure: Clear tracking of what's done vs pending
Recovery: Agent can resume from where it left off
Todo List Tools
Tool	Purpose
write_todos	Create a structured task list
update_todo	Mark tasks as complete/in-progress
list_todos	View current task state
from langchain_core.tools import tool
from typing import List, Optional
import json

# Simple in-memory todo storage for demonstration
# In production, Deep Agents use persistent storage
TODO_STORE = {}

@tool
def write_todos(todos: List[dict]) -> str:
    """Create a list of todos for tracking task progress.
    
    Args:
        todos: List of todo items, each with 'title' and optional 'description'
    
    Returns:
        Confirmation message with todo IDs
    """
    created = []
    for i, todo in enumerate(todos):
        todo_id = f"todo_{len(TODO_STORE) + i + 1}"
        TODO_STORE[todo_id] = {
            "id": todo_id,
            "title": todo.get("title", "Untitled"),
            "description": todo.get("description", ""),
            "status": "pending"
        }
        created.append(todo_id)
    return f"Created {len(created)} todos: {', '.join(created)}"

@tool
def update_todo(todo_id: str, status: Literal["pending", "in_progress", "completed"]) -> str:
    """Update the status of a todo item.
    
    Args:
        todo_id: The ID of the todo to update
        status: New status (pending, in_progress, completed)
    
    Returns:
        Confirmation message
    """
    if todo_id not in TODO_STORE:
        return f"Todo {todo_id} not found"
    TODO_STORE[todo_id]["status"] = status
    return f"Updated {todo_id} to {status}"

@tool
def list_todos() -> str:
    """List all todos with their current status.
    
    Returns:
        Formatted list of all todos
    """
    if not TODO_STORE:
        return "No todos found"
    
    result = []
    for todo_id, todo in TODO_STORE.items():
        status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        result.append(f"{emoji} [{todo_id}] {todo['title']} ({todo['status']})")
    return "\n".join(result)

print("Todo tools defined!")
Todo tools defined!
# Test the todo tools
TODO_STORE.clear()  # Reset for demo

# Create some wellness todos
result = write_todos.invoke({
    "todos": [
        {"title": "Assess current sleep patterns", "description": "Review user's sleep schedule and quality"},
        {"title": "Research sleep improvement strategies", "description": "Find evidence-based techniques"},
        {"title": "Create personalized sleep plan", "description": "Combine findings into actionable steps"},
    ]
})
print(result)
print("\nCurrent todos:")
print(list_todos.invoke({}))
Created 3 todos: todo_1, todo_3, todo_5

Current todos:
⬜ [todo_1] Assess current sleep patterns (pending)
⬜ [todo_3] Research sleep improvement strategies (pending)
⬜ [todo_5] Create personalized sleep plan (pending)
# Simulate progress
update_todo.invoke({"todo_id": "todo_1", "status": "completed"})
update_todo.invoke({"todo_id": "todo_2", "status": "in_progress"})

print("After updates:")
print(list_todos.invoke({}))
After updates:
✅ [todo_1] Assess current sleep patterns (completed)
⬜ [todo_3] Research sleep improvement strategies (pending)
⬜ [todo_5] Create personalized sleep plan (pending)
Task 4: Context Management with File Systems
The second key element is Context Management. Deep Agents use file systems to:

Offload large content - Store research, documents, and results to disk
Persist across sessions - Files survive beyond conversation context
Share between subagents - Subagents can read/write shared files
Prevent context overflow - Large tool results automatically saved to disk
Automatic Context Management
Deep Agents automatically handle context limits:

Large result offloading: Tool results >20k tokens → saved to disk
Proactive offloading: At 85% context capacity → agent saves state to disk
Summarization: Long conversations get summarized while preserving intent
File System Tools
Tool	Purpose
ls	List directory contents
read_file	Read file contents
write_file	Create/overwrite files
edit_file	Make targeted edits
import os
from pathlib import Path

# Create a workspace directory for our agent
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

@tool
def ls(path: str = ".") -> str:
    """List contents of a directory.
    
    Args:
        path: Directory path to list (default: current directory)
    
    Returns:
        List of files and directories
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"Directory not found: {path}"
    
    items = []
    for item in sorted(target.iterdir()):
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
        items.append(f"{prefix} {item.name}{size}")
    
    return "\n".join(items) if items else "(empty directory)"

@tool
def read_file(path: str) -> str:
    """Read contents of a file.
    
    Args:
        path: Path to the file to read
    
    Returns:
        File contents
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"File not found: {path}"
    return target.read_text()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (creates or overwrites).
    
    Args:
        path: Path to the file to write
        content: Content to write to the file
    
    Returns:
        Confirmation message
    """
    target = WORKSPACE / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {path}"

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing text.
    
    Args:
        path: Path to the file to edit
        old_text: Text to find and replace
        new_text: Replacement text
    
    Returns:
        Confirmation message
    """
    target = WORKSPACE / path
    if not target.exists():
        return f"File not found: {path}"
    
    content = target.read_text()
    if old_text not in content:
        return f"Text not found in {path}"
    
    new_content = content.replace(old_text, new_text, 1)
    target.write_text(new_content)
    return f"Updated {path}"

print("File system tools defined!")
print(f"Workspace: {WORKSPACE.absolute()}")
File system tools defined!
Workspace: /Users/kusumita/Documents/AI Makerspace/AIE9/07_Deep_Agents/workspace
# Test the file system tools
print("Current workspace contents:")
print(ls.invoke({"path": "."}))
Current workspace contents:
(empty directory)
# Create a research notes file
notes = """# Sleep Research Notes

## Key Findings
- Adults need 7-9 hours of sleep
- Consistent sleep schedule is important
- Blue light affects melatonin production

## TODO
- [ ] Review individual user needs
- [ ] Create personalized recommendations
"""

result = write_file.invoke({"path": "research/sleep_notes.md", "content": notes})
print(result)

# Verify it was created
print("\nResearch directory:")
print(ls.invoke({"path": "research"}))
Wrote 242 characters to research/sleep_notes.md

Research directory:
[FILE] sleep_notes.md (242 bytes)
# Read and edit the file
print("File contents:")
print(read_file.invoke({"path": "research/sleep_notes.md"}))
File contents:
# Sleep Research Notes

## Key Findings
- Adults need 7-9 hours of sleep
- Consistent sleep schedule is important
- Blue light affects melatonin production

## TODO
- [ ] Review individual user needs
- [ ] Create personalized recommendations

Task 5: Basic Deep Agent
Now let's create a basic Deep Agent using the deepagents package. This combines:

Planning (todo lists)
Context management (file system)
A capable LLM backbone
Configuring the FilesystemBackend
Deep Agents come with built-in file tools (ls, read_file, write_file, edit_file). To control where files are stored, we configure a FilesystemBackend:

from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(
    root_dir="/path/to/workspace",
    virtual_mode=True  # REQUIRED to actually sandbox files!
)
Critical: virtual_mode=True

Without virtual_mode=True, agents can still write anywhere on the filesystem!
The root_dir alone does NOT restrict file access
virtual_mode=True blocks paths with .., ~, and absolute paths outside root
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

# Configure the filesystem backend to use our workspace directory
# IMPORTANT: virtual_mode=True is required to actually restrict paths to root_dir
# Without it, agents can still write anywhere on the filesystem!
workspace_path = Path("workspace").absolute()
filesystem_backend = FilesystemBackend(
    root_dir=str(workspace_path),
    virtual_mode=True  # This is required to sandbox file operations!
)

# Combine our custom tools (for todo tracking)
# Note: Deep Agents has built-in file tools (ls, read_file, write_file, edit_file)
# that will use the configured FilesystemBackend
custom_tools = [
    write_todos,
    update_todo,
    list_todos,
]

# Create a basic Deep Agent
wellness_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=custom_tools,
    backend=filesystem_backend,  # Configure where files are stored
    system_prompt="""You are a Personal Wellness Assistant that helps users improve their health.

When given a complex task:
1. First, create a todo list to track your progress
2. Work through each task, updating status as you go
3. Save important findings to files for reference
4. Provide a clear summary when complete

Be thorough but concise. Always explain your reasoning."""
)

print(f"Basic Deep Agent created!")
print(f"File operations sandboxed to: {workspace_path}")
Basic Deep Agent created!
File operations sandboxed to: /Users/kusumita/Documents/AI Makerspace/AIE9/07_Deep_Agents/workspace
# Reset todo store for fresh demo
TODO_STORE.clear()

# Test with a multi-step wellness task
result = wellness_agent.invoke({
    "messages": [{
        "role": "user",
        "content": """I want to improve my sleep quality. I currently:
- Go to bed at inconsistent times (10pm-1am)
- Use my phone in bed
- Often feel tired in the morning

Please create a personalized sleep improvement plan for me and save it to a file."""
    }]
})

print("Agent response:")
print(result["messages"][-1].content)
Agent response:
Perfect! I've created a comprehensive, personalized sleep improvement plan and saved it to `/personalized_sleep_improvement_plan.md`. 

## Summary of Your Sleep Plan:

**Your main issues addressed:**
- **Inconsistent bedtime** → Fixed schedule: 10:30 PM bedtime, 6:30 AM wake time
- **Phone use in bed** → Complete phone removal from bedroom + 1-hour screen curfew
- **Morning fatigue** → Strategic morning light exposure + optimized sleep quality

**The plan is structured in 3 phases over 6 weeks:**

1. **Weeks 1-2**: Foundation - establishing consistent sleep schedule and removing phones
2. **Weeks 3-4**: Light optimization - managing blue light and morning light exposure  
3. **Weeks 5-6**: Fine-tuning and troubleshooting remaining issues

**Key immediate changes to start today:**
- Choose 10:30 PM as your consistent bedtime (7 days/week)
- Remove your phone from the bedroom completely
- Create a 9:30 PM screen curfew
- Get bright light exposure within 30 minutes of waking

The plan includes daily schedules, troubleshooting guides, success tracking metrics, and clear guidance on when to seek professional help if needed. Research shows that following this systematic approach for 6-8 weeks typically results in significantly improved sleep quality and morning energy levels.

Would you like me to clarify any part of the plan or help you prepare for starting your sleep transformation?
# Check what the agent created
print("Todo list after task:")
print(list_todos.invoke({}))

print("\n" + "="*50)
print("\nWorkspace contents:")
# List files in the workspace directory
for f in sorted(WORKSPACE.iterdir()):
    if f.is_file():
        print(f"  [FILE] {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"  [DIR] {f.name}/")
Todo list after task:
✅ [todo_1] Analyze current sleep issues (completed)
✅ [todo_3] Research evidence-based sleep improvement strategies (completed)
✅ [todo_5] Research circadian rhythm science and consistent sleep schedules (completed)
✅ [todo_7] Save plan to file (completed)
✅ [todo_6] Analyze screen time and blue light impact on sleep (completed)
✅ [todo_8] Investigate blue light mitigation strategies (completed)
✅ [todo_10] Research morning fatigue causes and solutions (completed)
✅ [todo_12] Compile practical sleep hygiene implementation strategies (completed)
✅ [todo_14] Create comprehensive research summary document (completed)

==================================================

Workspace contents:
  [FILE] personalized_sleep_improvement_plan.md (5989 bytes)
  [DIR] research/
  [FILE] sleep_research_summary.md (9833 bytes)
❓ Question #1:
What are the trade-offs of using todo lists for planning? Consider:

When might explicit planning overhead slow things down?
How granular should todo items be?
What happens if the agent creates todos but never completes them?
Answer:
Using todo lists for planning improves structure and reliability in long-horizon agent workflows, but it also introduces some execution overhead. The main benefit is that the agent no longer has to keep all task state inside the conversation context. Instead, the plan becomes a persistent, visible artifact that both the user and the agent can track. This improves transparency, recovery after interruptions, and coordination across multiple steps or subagents.

However, explicit planning can slow things down when the task is simple or requires only one or two actions. In those cases, creating, updating, and reading todo items adds unnecessary steps and extra tool calls. Planning can also introduce latency if the agent repeatedly re-plans instead of executing. In short, todo lists are most valuable for complex, multi-step work, but they become process overhead for straightforward tasks.

The level of granularity is also important. Todo items should represent meaningful milestones rather than tiny micro-actions. If they are too granular, the agent spends more time managing tasks than making progress, and the system accumulates noise. If they are too coarse, it becomes difficult to track progress or recover from partial failures. A good rule is that each todo should capture a logical phase such as “research strategies” or “create final plan,” not individual clicks or thoughts.

Another risk appears when the agent creates todos but never completes them. This leads to stale or misleading state where tasks remain permanently “pending,” which reduces user trust and makes future sessions harder to resume correctly. Over time, unfinished tasks accumulate and the plan loses meaning.

Common mitigations include: - Periodically reconciling or cleaning stale todos - Requiring status updates after major steps - Limiting the number of open tasks - Using a supervisor or checkpoint step to ensure completion

Overall, todo lists turn implicit reasoning into explicit execution state. This greatly improves reliability, visibility, and long-horizon task management, but it must be balanced with lightweight planning and active maintenance to avoid slowing the system down.

❓ Question #2:
How would you design a context management strategy for a wellness agent that:

Needs to reference a large health document (16KB)
Tracks user metrics over time
Must remember user conditions (allergies, medications) for safety
What goes in files vs. in the prompt? What should never be offloaded?

Answer:
We can design the context management using three layers so the agent stays safe while avoiding context overload.

i. The large health document (16KB) should be stored in files and accessed through retrieval when needed, instead of putting it into the prompt. Only the relevant sections should be pulled into the context for each question. This keeps the conversation window small and prevents token waste.

ii. User metrics over time (sleep, steps, mood, weight, etc.) should also be stored in files or structured storage (CSV/JSON) and summarized periodically. The prompt should include only a short recent snapshot (for example, last 7-day averages), not the full history.

iii. User conditions like allergies and medications must be stored in long-term memory as structured data, and a short safety summary should always be injected into the prompt so the agent never forgets critical constraints.

What goes where: - Prompt: safety constraints, current goals, recent metric summary - Files: large health documents, detailed notes, full metric history - Long-term memory: stable profile data (conditions, preferences, history)

What should never be offloaded: Safety critical information such as allergies, medications, major medical conditions, and emergency boundaries. These must always remain visible in the active context to avoid harmful recommendations.

🏗️ Activity #1: Build a Research Agent
Build a Deep Agent that can research a wellness topic and produce a structured report.

Requirements:
Create todos for the research process
Read from the HealthWellnessGuide.txt in the data folder
Save findings to a structured markdown file
Update todo status as tasks complete
Test prompt:
"Research stress management techniques and create a comprehensive guide with at least 5 evidence-based strategies."

Step 1: Todo tools (planning)

from pathlib import Path
from typing import List, Literal
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# ------------------------------------------------------------
# Step 1: Todo tools (planning)
# ------------------------------------------------------------
TODO_STORE = {}

@tool
def write_todos(todos: List[dict]) -> str:
    """Create todos. Each item: {'title': str, 'description': str (optional)}"""
    created = []
    for t in todos:
        tid = f"todo_{len(TODO_STORE) + 1}"
        TODO_STORE[tid] = {
            "id": tid,
            "title": t.get("title", "Untitled"),
            "description": t.get("description", ""),
            "status": "pending",
        }
        created.append(tid)
    return f"Created {len(created)} todos: {', '.join(created)}"

@tool
def update_todo(todo_id: str, status: Literal["pending", "in_progress", "completed"]) -> str:
    """Update a todo status."""
    if todo_id not in TODO_STORE:
        return f"Todo not found: {todo_id}"
    TODO_STORE[todo_id]["status"] = status
    return f"Updated {todo_id} -> {status}"

@tool
def list_todos() -> str:
    """List all todos."""
    if not TODO_STORE:
        return "No todos found"
    icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    lines = []
    for tid, t in TODO_STORE.items():
        lines.append(f"{icon.get(t['status'], '❓')} [{tid}] {t['title']} ({t['status']})")
    return "\n".join(lines)
Step 2: Tool to read the source document from /data

# ------------------------------------------------------------
# Step 2: Tool to read the source document from /data
# ------------------------------------------------------------
@tool
def read_health_wellness_guide() -> str:
    """Read HealthWellnessGuide.txt from the data folder (required source)."""
    p = Path("data/HealthWellnessGuide.txt")
    if not p.exists():
        return f"ERROR: Missing file: {p.resolve()}"
    return p.read_text(encoding="utf-8", errors="ignore")
Step 3: Create the Deep Agent (with sandboxed workspace)

# ------------------------------------------------------------
# Step 3: Create the Deep Agent (with sandboxed workspace)
# ------------------------------------------------------------
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

filesystem_backend = FilesystemBackend(
    root_dir=str(WORKSPACE.absolute()),
    virtual_mode=True  # IMPORTANT: sandbox file operations to workspace/
)

research_agent = create_deep_agent(
    model=init_chat_model("openai:gpt-4o-mini"),  # use OpenAI to avoid Anthropic credit errors
    tools=[
        write_todos, update_todo, list_todos,
        read_health_wellness_guide
    ],
    backend=filesystem_backend,  # enables built-in file tools: ls/read_file/write_file/edit_file (inside workspace)
    system_prompt="""
You are a Wellness Research Agent.

PLANNING IS REQUIRED.
- Before doing ANY research or writing, you MUST call write_todos() with 5–7 todos.
- You MUST set at least one todo to in_progress using update_todo().
- As you complete each major step, you MUST mark the relevant todo completed using update_todo().
- If you have not created todos, STOP and create them first.

You MUST follow this process:
1) Create todos (5–7).
2) Read the ONLY allowed source using read_health_wellness_guide().
3) Produce a structured report based ONLY on that source (do not invent evidence).
4) Save the report to: reports/stress_management_guide.md (inside the workspace).
5) Update todos as each step completes.
6) In your final response, you MUST show the todo list by calling list_todos() and include its output.

Report format (markdown):
- Title
- Summary (5-8 lines)
- Evidence-based strategies (at least 5):
  For each: What it is | Why it helps | How to do it | Practical tips
- 7-day starter plan
- Safety & limitations
"""

)
Step 4: Test prompt

# ------------------------------------------------------------
# Step 4: Test prompt
# ------------------------------------------------------------
TODO_STORE.clear()

test_prompt = """
REQUIREMENTS (must follow):
1) First create a todo list using write_todos() (5–7 todos).
2) Then read the source using read_health_wellness_guide().
3) Save the final report to reports/stress_management_guide.md.
4) Update todos as you complete steps.
5) At the end, show the todos using list_todos().

Task:
Research stress management techniques and create a comprehensive guide with at least 5 evidence-based strategies.
"""


result = research_agent.invoke({
    "messages": [{"role": "user", "content": test_prompt}]
})

print(result["messages"][-1].content)

print("\n--- TODOS ---")
print(list_todos.invoke({}))

print("\n--- OUTPUT FILE CHECK (workspace/reports) ---")
reports_dir = WORKSPACE / "reports"
if reports_dir.exists():
    for f in sorted(reports_dir.iterdir()):
        if f.is_file():
            print(f"- {f.name} ({f.stat().st_size} bytes)")
else:
    print("No reports/ folder found yet (agent may not have written the file).")
Here is the summary of the completed tasks:

### Todo List
- ✅ [todo_1] Create todo list for research on stress management techniques (completed)
- 🔄 [todo_2] Read the Health Wellness Guide (in_progress)
- 🔄 [todo_3] Draft the stress management guide (in_progress)
- ✅ [todo_4] Finalize the guide (completed)
- ✅ [todo_5] Save the guide to the reports directory (completed)
- ✅ [todo_6] Update todos after each step (completed)
- ✅ [todo_7] Display the final todo list (completed)

### Report Overview
A comprehensive guide on stress management with various evidence-based strategies has been successfully created and saved. If you have any further requests or need additional information, feel free to ask!

--- TODOS ---
✅ [todo_1] Create todo list for research on stress management techniques (completed)
🔄 [todo_2] Read the Health Wellness Guide (in_progress)
🔄 [todo_3] Draft the stress management guide (in_progress)
✅ [todo_4] Finalize the guide (completed)
✅ [todo_5] Save the guide to the reports directory (completed)
✅ [todo_6] Update todos after each step (completed)
✅ [todo_7] Display the final todo list (completed)

--- OUTPUT FILE CHECK (workspace/reports) ---
- stress_management_guide.md (11481683 bytes)
🤝 Breakout Room #2
Advanced Features & Integration
Task 6: Subagent Spawning
The third key element is Subagent Spawning. This allows a Deep Agent to delegate tasks to specialized subagents.

Why Subagents?
Context Isolation: Each subagent has its own context window, preventing bloat
Specialization: Different subagents can have different tools/prompts
Parallelism: Multiple subagents can work simultaneously
Cost Optimization: Use cheaper models for simpler subtasks
How Subagents Work
Main Agent
    ├── task("Research sleep science", model="gpt-4o-mini")
    │       └── Returns: Summary of findings
    │
    ├── task("Analyze user's sleep data", tools=[analyze_tool])
    │       └── Returns: Analysis results
    │
    └── task("Write recommendations", system_prompt="Be concise")
            └── Returns: Final recommendations
Key benefit: The main agent only receives summaries, not all the intermediate context!

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

# Define specialized subagent configurations
# Note: Subagents inherit the backend from the parent agent
research_subagent = {
    "name": "research-agent",
    "description": "Use this agent to research wellness topics in depth. It can read documents and synthesize information.",
    "system_prompt": """You are a wellness research specialist. Your job is to:
1. Find relevant information in provided documents
2. Synthesize findings into clear summaries
3. Cite sources when possible

Be thorough but concise. Focus on evidence-based information.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "openai:gpt-4o-mini",  # Cheaper model for research
}

writing_subagent = {
    "name": "writing-agent",
    "description": "Use this agent to create well-structured documents, plans, and guides.",
    "system_prompt": """You are a wellness content writer. Your job is to:
1. Take research findings and turn them into clear, actionable content
2. Structure information for easy understanding
3. Use formatting (headers, bullets, etc.) effectively

Write in a supportive, encouraging tone.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "anthropic:claude-sonnet-4-20250514",
}

print("Subagent configurations defined!")
Subagent configurations defined!
# Create a coordinator agent that can spawn subagents
coordinator_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=[write_todos, update_todo, list_todos],
    backend=filesystem_backend,  # Use the same backend - subagents inherit it
    subagents=[research_subagent, writing_subagent],
    system_prompt="""You are a Wellness Project Coordinator. Your role is to:
1. Break down complex wellness requests into subtasks
2. Delegate research to the research-agent
3. Delegate content creation to the writing-agent
4. Coordinate the overall workflow using todos

Use subagents for specialized work rather than doing everything yourself.
This keeps the work organized and the results high-quality."""
)

print("Coordinator agent created with subagent capabilities!")
Coordinator agent created with subagent capabilities!
# Reset for demo
TODO_STORE.clear()

# Test the coordinator with a complex task
result = coordinator_agent.invoke({
    "messages": [{
        "role": "user",
        "content": """Create a comprehensive morning routine guide for better energy.
        
The guide should:
1. Research the science behind morning routines
2. Include practical steps for exercise, nutrition, and mindset
3. Be saved as a well-formatted markdown file"""
    }]
})

print("Coordinator response:")
print(result["messages"][-1].content)
Coordinator response:
## ✅ Project Complete!

I've successfully created your comprehensive morning routine guide for better energy! Here's what I delivered:

### 📋 **What Was Accomplished:**

1. **✅ Scientific Research** - Thoroughly researched the science behind morning routines, covering:
   - Circadian rhythm impacts on energy
   - Cortisol awakening response
   - Exercise timing research
   - Nutrition and meal timing studies
   - Sleep-wake cycle optimization
   - Mindfulness and mental preparation benefits
   - Light exposure effects
   - Hydration science

2. **✅ Comprehensive Guide Creation** - Created a detailed, well-structured guide with:
   - Engaging introduction explaining why morning routines matter
   - Science-backed foundation in accessible language
   - Four practical pillars: Movement, Nutrition, Mindset, and Light/Environment
   - Customization options for different lifestyles and chronotypes
   - Common mistakes and troubleshooting solutions
   - Sample routines for 15, 30, and 60+ minute timeframes
   - Step-by-step implementation strategy

3. **✅ Professional Formatting** - Saved as a beautifully formatted markdown file at:
   `/comprehensive_morning_routine_guide.md`

### 🌟 **Key Features of Your Guide:**

- **Science-Based**: Every recommendation is backed by research on circadian rhythms, hormones, and energy optimization
- **Practical**: Includes specific timing, foods, exercises, and mindset practices
- **Customizable**: Adaptable for early birds, night owls, busy parents, professionals, and more
- **Actionable**: Clear checklists and step-by-step routines you can implement immediately
- **Sustainable**: Focus on building habits gradually rather than overwhelming changes

The guide transforms complex scientific research into an accessible, encouraging resource that anyone can use to boost their morning energy and improve their entire day. You now have a comprehensive, evidence-based morning routine guide ready to use!
# Check the results
print("Final todo status:")
print(list_todos.invoke({}))

print("\nGenerated files in workspace:")
for f in sorted(WORKSPACE.iterdir()):
    if f.is_file():
        print(f"  [FILE] {f.name} ({f.stat().st_size} bytes)")
    elif f.is_dir():
        print(f"  [DIR] {f.name}/")
Final todo status:
✅ [todo_1] Research the science behind morning routines (completed)
✅ [todo_3] Create comprehensive morning routine guide (completed)
✅ [todo_5] Save the guide as a markdown file (completed)

Generated files in workspace:
  [FILE] .gitkeep (0 bytes)
  [FILE] comprehensive_morning_routine_guide.md (25117 bytes)
  [FILE] morning_routine_guide.md (54068 bytes)
  [FILE] personalized_sleep_improvement_plan.md (5292 bytes)
  [DIR] research/
  [FILE] sleep_hygiene_research.md (15886 bytes)
Task 7: Long-term Memory Integration
The fourth key element is Long-term Memory. Deep Agents integrate with LangGraph's Store for persistent memory across sessions.

Memory Types in Deep Agents
Type	Scope	Use Case
Thread Memory	Single conversation	Current session context
User Memory	Across threads, per user	User preferences, history
Shared Memory	Across all users	Common knowledge, learned patterns
Integration with LangGraph Store
Deep Agents can use the same InMemoryStore (or PostgresStore) we learned in Session 6:

from langgraph.store.memory import InMemoryStore

# Create a memory store
memory_store = InMemoryStore()

# Store user profile
user_id = "user_alex"
profile_namespace = (user_id, "profile")

memory_store.put(profile_namespace, "name", {"value": "Alex"})
memory_store.put(profile_namespace, "goals", {
    "primary": "improve energy levels",
    "secondary": "better sleep"
})
memory_store.put(profile_namespace, "conditions", {
    "dietary": ["vegetarian"],
    "medical": ["mild anxiety"]
})
memory_store.put(profile_namespace, "preferences", {
    "exercise_time": "morning",
    "communication_style": "detailed"
})

print(f"Stored profile for {user_id}")

# Retrieve and display
for item in memory_store.search(profile_namespace):
    print(f"  {item.key}: {item.value}")
Stored profile for user_alex
  name: {'value': 'Alex'}
  goals: {'primary': 'improve energy levels', 'secondary': 'better sleep'}
  conditions: {'dietary': ['vegetarian'], 'medical': ['mild anxiety']}
  preferences: {'exercise_time': 'morning', 'communication_style': 'detailed'}
# Create memory-aware tools
from langgraph.store.base import BaseStore

@tool
def get_user_profile(user_id: str) -> str:
    """Retrieve a user's wellness profile from long-term memory.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        User profile as formatted text
    """
    namespace = (user_id, "profile")
    items = list(memory_store.search(namespace))
    
    if not items:
        return f"No profile found for {user_id}"
    
    result = [f"Profile for {user_id}:"]
    for item in items:
        result.append(f"  {item.key}: {item.value}")
    return "\n".join(result)

@tool
def save_user_preference(user_id: str, key: str, value: str) -> str:
    """Save a user preference to long-term memory.
    
    Args:
        user_id: The user's unique identifier
        key: The preference key
        value: The preference value
    
    Returns:
        Confirmation message
    """
    namespace = (user_id, "preferences")
    memory_store.put(namespace, key, {"value": value})
    return f"Saved preference '{key}' for {user_id}"

print("Memory tools defined!")
Memory tools defined!
# Create a memory-enhanced agent
memory_tools = [
    get_user_profile,
    save_user_preference,
    write_todos,
    update_todo,
    list_todos,
]

memory_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=memory_tools,
    backend=filesystem_backend,  # Use workspace for file operations
    system_prompt="""You are a Personal Wellness Assistant with long-term memory.

At the start of each conversation:
1. Check the user's profile to understand their goals and conditions
2. Personalize all advice based on their profile
3. Save any new preferences they mention

Always reference stored information to show you remember the user."""
)

print("Memory-enhanced agent created!")
Memory-enhanced agent created!
# Test the memory agent
TODO_STORE.clear()

result = memory_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Hi! My user_id is user_alex. What exercise routine would you recommend for me?"
    }]
})

print("Agent response:")
print(result["messages"][-1].content)
Agent response:
Hello Alex! Based on your profile, I can see your primary goal is to improve energy levels with better sleep as a secondary goal, and you prefer morning exercise. You also experience mild anxiety, which I'll keep in mind for my recommendations.

Here's a personalized exercise routine that should help boost your energy and improve your sleep quality:

## **Morning Energy-Boosting Routine (30-45 minutes)**

### **Monday, Wednesday, Friday - Energizing Cardio & Strength**
- **5-10 minutes**: Gentle dynamic warm-up (arm circles, leg swings, light stretching)
- **20-25 minutes**: Moderate cardio that won't spike anxiety
  - Brisk walking or light jogging
  - Swimming (excellent for anxiety management)
  - Cycling at a comfortable pace
- **10-15 minutes**: Bodyweight strength training
  - Squats, push-ups (modified if needed), planks
  - Focus on controlled movements for mental calmness

### **Tuesday, Thursday - Yoga & Mindful Movement**
- **30-40 minutes**: Morning yoga flow
  - Sun salutations to energize
  - Poses that improve circulation and reduce anxiety
  - End with brief meditation (3-5 minutes)

### **Saturday - Active Recovery**
- **20-30 minutes**: Gentle nature walk or easy bike ride
- Focus on being outdoors for mood and energy benefits

### **Sunday - Rest or Gentle Stretching**
- Optional 15-20 minutes of restorative stretching

## **Why This Works for You:**
- **Morning timing** aligns with your preference and helps establish circadian rhythm for better sleep
- **Moderate intensity** boosts energy without overwhelming your nervous system
- **Yoga and mindful movement** help manage mild anxiety while improving flexibility
- **Consistent routine** helps regulate sleep-wake cycles

Would you like me to save any specific preferences about this routine, or would you like me to adjust anything based on your current fitness level or time constraints?
Task 8: Skills - On-Demand Capabilities
Skills are a powerful feature for progressive capability disclosure. Instead of loading all tools upfront, agents can load specialized capabilities on demand.

Why Skills?
Context Efficiency: Don't waste context on unused tool descriptions
Specialization: Skills can include detailed instructions for specific tasks
Modularity: Easy to add/remove capabilities
Discoverability: Agent can browse available skills
SKILL.md Format
Skills are defined in markdown files with YAML frontmatter:

---
name: skill-name
description: What this skill does
version: 1.0.0
tools:
  - tool1
  - tool2
---

# Skill Instructions

Detailed steps for how to use this skill...
# Let's look at the skills we created
skills_dir = Path("skills")

print("Available skills:")
for skill_dir in skills_dir.iterdir():
    if skill_dir.is_dir():
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            content = skill_file.read_text()
            # Extract name and description from frontmatter
            lines = content.split("\n")
            name = ""
            desc = ""
            for line in lines:
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
            print(f"  - {name}: {desc}")
Available skills:
  - meal-planning: Create personalized meal plans based on dietary needs and preferences
  - wellness-assessment: Assess user wellness goals and create personalized recommendations
# Read the wellness-assessment skill
skill_content = Path("skills/wellness-assessment/SKILL.md").read_text()
print(skill_content)
---
name: wellness-assessment
description: Assess user wellness goals and create personalized recommendations
version: 1.0.0
tools:
  - read_file
  - write_file
---

# Wellness Assessment Skill

You are conducting a comprehensive wellness assessment. Follow these steps:

## Step 1: Gather Information
Ask the user about:
- Current health goals (weight, fitness, stress, sleep)
- Any medical conditions or limitations
- Current exercise routine (or lack thereof)
- Dietary preferences and restrictions
- Sleep patterns and quality
- Stress levels and sources

## Step 2: Analyze Responses
Review the user's answers and identify:
- Primary wellness priority
- Secondary goals
- Potential barriers to success
- Existing healthy habits to build on

## Step 3: Create Assessment Report
Write a wellness assessment report to `workspace/wellness_assessment.md` containing:
- Summary of current wellness state
- Identified strengths
- Areas for improvement
- Recommended focus areas (prioritized)
- Suggested next steps

## Step 4: Provide Recommendations
Based on the assessment, provide:
- 3 immediate action items (can start today)
- 3 short-term goals (1-2 weeks)
- 3 long-term goals (1-3 months)

## Output Format
Always save results to the workspace directory and provide a clear summary to the user.

# Create a skill-aware tool
@tool
def load_skill(skill_name: str) -> str:
    """Load a skill's instructions for a specialized task.
    
    Available skills:
    - wellness-assessment: Assess user wellness and create recommendations
    - meal-planning: Create personalized meal plans
    
    Args:
        skill_name: Name of the skill to load
    
    Returns:
        Skill instructions
    """
    skill_path = Path(f"skills/{skill_name}/SKILL.md")
    if not skill_path.exists():
        available = [d.name for d in Path("skills").iterdir() if d.is_dir()]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"
    
    return skill_path.read_text()

print("Skill loader defined!")
Skill loader defined!
# Create an agent that can load and use skills
skill_agent = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=[
        load_skill,
        write_todos,
        update_todo,
        list_todos,
    ],
    backend=filesystem_backend,  # Use workspace for file operations
    system_prompt="""You are a wellness assistant with access to specialized skills.

When a user asks for something that matches a skill:
1. Load the appropriate skill using load_skill()
2. Follow the skill's instructions carefully
3. Save outputs as specified in the skill

Available skills:
- wellness-assessment: For comprehensive wellness evaluations
- meal-planning: For creating personalized meal plans

If no skill matches, use your general wellness knowledge."""
)

print("Skill-aware agent created!")
Skill-aware agent created!
# Test with a skill-appropriate request
TODO_STORE.clear()

result = skill_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "I'd like a wellness assessment. I'm a 35-year-old office worker who sits most of the day, has trouble sleeping, and wants to lose 15 pounds. I'm vegetarian and have no major health conditions."
    }]
})

print("Agent response:")
print(result["messages"][-1].content)
Agent response:
## Your Personalized Wellness Assessment Summary

Based on your profile as a 35-year-old office worker, I've identified that your challenges are interconnected - sedentary work, sleep difficulties, and weight management all influence each other. Here's your prioritized action plan:

### 🎯 **Immediate Actions (Start Today):**
1. **Set hourly movement reminders** - Just 2-3 minutes of standing/stretching every hour
2. **Establish consistent sleep/wake times** - Your body craves routine for better sleep
3. **Pre-meal hydration** - Drink water before eating to support metabolism and appetite control

### 📅 **Next 1-2 Weeks:**
1. **Daily 15-20 minute walks** - Can be broken into smaller segments (great for office workers!)
2. **Screen curfew 1 hour before bed** - This alone can dramatically improve sleep quality
3. **Plan balanced vegetarian meals** - Ensure adequate protein at each meal for weight management

### 🏆 **1-3 Month Goals:**
1. **3-4 weekly exercise sessions** - Mix of cardio and strength training
2. **6-8 lbs sustainable weight loss** - About 1-2 lbs per week
3. **Consistent 7-8 hours quality sleep** - The foundation of all other improvements

### 💡 **Key Insight:**
Your biggest opportunity is **breaking the sedentary cycle**. Even small movement breaks can kickstart your metabolism, improve energy, and help with both weight loss and sleep quality.

The full detailed assessment has been saved to your workspace. Would you like me to help you create a specific meal plan for your vegetarian diet, or would you prefer to focus on developing a movement routine that works with your office schedule?
Task 9: Using deepagents-cli
The deepagents-cli provides an interactive terminal interface for working with Deep Agents.

Installation
uv pip install deepagents-cli
# or
pip install deepagents-cli
Key Features
Feature	Description
Interactive Sessions	Chat with your agent in the terminal
Conversation Resume	Pick up where you left off
Human-in-the-Loop	Approve or reject agent actions
File System Access	Agent can read/write to your filesystem
Remote Sandboxing	Run in isolated Docker containers
Basic Usage
# Start an interactive session
deepagents

# Resume a previous conversation
deepagents --resume

# Use a specific model
deepagents --model openai:gpt-4o

# Enable human-in-the-loop approval
deepagents --approval-mode full
Example Session
$ deepagents

Welcome to Deep Agents CLI!

You: Create a 7-day meal plan for a vegetarian athlete

Agent: I'll create a comprehensive meal plan for you. Let me:
1. Research vegetarian athlete nutrition needs
2. Design balanced daily menus
3. Save the plan to a file

[Agent uses tools...]

Agent: I've created your meal plan! You can find it at:
workspace/vegetarian_athlete_meal_plan.md

You: /exit
# Check if CLI is installed
import subprocess

try:
    result = subprocess.run(["deepagents", "--version"], capture_output=True, text=True)
    print(f"deepagents-cli version: {result.stdout.strip()}")
except FileNotFoundError:
    print("deepagents-cli not installed. Install with:")
    print("  uv pip install deepagents-cli")
    print("  # or")
    print("  pip install deepagents-cli")
deepagents-cli not installed. Install with:
  uv pip install deepagents-cli
  # or
  pip install deepagents-cli
Try It Yourself!
After installing the CLI, try these commands in your terminal:

# Basic interactive session
deepagents

# With a specific working directory
deepagents --workdir ./workspace

# See all options
deepagents --help
Sample prompts to try:

"Create a weekly workout plan and save it to a file"
"Research the health benefits of meditation and summarize in a report"
"Analyze my current diet and suggest improvements" (then provide details)
Task 10: Building a Complete Deep Agent System
Now let's bring together all four elements to build a comprehensive "Wellness Coach" system:

Planning: Track multi-week wellness programs
Context Management: Store session notes and progress
Subagent Spawning: Delegate to specialists (exercise, nutrition, mindfulness)
Long-term Memory: Remember user preferences and history
# Define specialized wellness subagents
# Subagents inherit the backend from the parent, so they use the same workspace
exercise_specialist = {
    "name": "exercise-specialist",
    "description": "Expert in exercise science, workout programming, and physical fitness. Use for exercise-related questions and plan creation.",
    "system_prompt": """You are an exercise specialist with expertise in:
- Workout programming for different fitness levels
- Exercise form and safety
- Progressive overload principles
- Recovery and injury prevention

Always consider the user's fitness level and any physical limitations.
Provide clear, actionable exercise instructions.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "openai:gpt-4o-mini",
}

nutrition_specialist = {
    "name": "nutrition-specialist",
    "description": "Expert in nutrition science, meal planning, and dietary optimization. Use for food-related questions and meal plans.",
    "system_prompt": """You are a nutrition specialist with expertise in:
- Macro and micronutrient balance
- Meal planning and preparation
- Dietary restrictions and alternatives
- Nutrition timing for performance

Always respect dietary restrictions and preferences.
Focus on practical, achievable meal suggestions.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "openai:gpt-4o-mini",
}

mindfulness_specialist = {
    "name": "mindfulness-specialist",
    "description": "Expert in stress management, sleep optimization, and mental wellness. Use for stress, sleep, and mental health questions.",
    "system_prompt": """You are a mindfulness and mental wellness specialist with expertise in:
- Stress reduction techniques
- Sleep hygiene and optimization
- Meditation and breathing exercises
- Work-life balance strategies

Be supportive and non-judgmental.
Provide practical techniques that can be implemented immediately.""",
    "tools": [],  # Uses built-in file tools from backend
    "model": "openai:gpt-4o-mini",
}

print("Specialist subagents defined!")
Specialist subagents defined!
# Create the Wellness Coach coordinator
wellness_coach = create_deep_agent(
    model=init_chat_model("anthropic:claude-sonnet-4-20250514"),
    tools=[
        # Planning
        write_todos,
        update_todo,
        list_todos,
        # Long-term Memory
        get_user_profile,
        save_user_preference,
        # Skills
        load_skill,
    ],
    backend=filesystem_backend,  # All file ops go to workspace
    subagents=[exercise_specialist, nutrition_specialist, mindfulness_specialist],
    system_prompt="""You are a Personal Wellness Coach that coordinates comprehensive wellness programs.

## Your Role
- Understand each user's unique goals, constraints, and preferences
- Create personalized, multi-week wellness programs
- Coordinate between exercise, nutrition, and mindfulness specialists
- Track progress and adapt recommendations

## Workflow
1. **Initial Assessment**: Get user profile and understand their situation
2. **Planning**: Create a todo list for the program components
3. **Delegation**: Use specialists for domain-specific content:
   - exercise-specialist: Workout plans and fitness guidance
   - nutrition-specialist: Meal plans and dietary advice
   - mindfulness-specialist: Stress and sleep optimization
4. **Integration**: Combine specialist outputs into a cohesive program
5. **Documentation**: Save all plans and recommendations to files

## Important
- Always check user profile first for context
- Respect any medical conditions or dietary restrictions
- Provide clear, actionable recommendations
- Save progress to files so users can reference later"""
)

print("Wellness Coach created with all 4 Deep Agent elements!")
Wellness Coach created with all 4 Deep Agent elements!
# Test the complete system
TODO_STORE.clear()

result = wellness_coach.invoke({
    "messages": [{
        "role": "user",
        "content": """Hi! My user_id is user_alex. I'd like you to create a 2-week wellness program for me.

I want to focus on:
1. Building a consistent exercise routine (I can exercise 3x per week for 30 mins)
2. Improving my diet (remember I'm vegetarian)
3. Better managing my work stress and improving my sleep

Please create comprehensive plans for each area and save them as separate files I can reference."""
    }]
})

print("Wellness Coach response:")
print(result["messages"][-1].content)
Wellness Coach response:
## Your Complete 2-Week Wellness Program is Ready! 🎉

Alex, I've created a comprehensive wellness program perfectly tailored to your needs and saved it in 4 detailed reference files:

### 📋 **Your Program Files:**
1. **`/alex_2week_exercise_plan.md`** - Complete workout routines (3x/week, 30 min each)
2. **`/alex_2week_nutrition_plan.md`** - Full vegetarian meal plans with prep guides  
3. **`/alex_2week_stress_sleep_plan.md`** - Daily stress management and sleep optimization
4. **`/alex_2week_integrated_wellness_program.md`** - Master schedule combining everything

### 🌟 **What Makes Your Program Special:**

**Perfectly Personalized:**
- Respects your vegetarian diet and mild anxiety
- 3x/week morning exercise schedule (as you prefer)
- Foods specifically chosen to boost energy and support mental wellness
- Stress techniques designed for work situations

**Scientifically Integrated:**
- Exercise timing supports better sleep
- Nutrition fuels your workouts and reduces anxiety
- Stress management enhances both exercise performance and recovery
- All components work together to maximize energy improvements

**Practical & Sustainable:**
- 30-minute workouts that fit your schedule
- Simple meal prep strategies for busy weeks
- Quick stress-relief techniques you can use anywhere
- Progressive difficulty that builds confidence

### 🗓️ **Your Next Steps:**
1. **Review the integrated program** first for the big picture
2. **Start Monday** with Day 1 of your program
3. **Use the individual files** for detailed daily guidance
4. **Track your progress** using the suggested check-ins

### 💡 **Key Success Tips:**
- **Consistency over perfection** - every healthy choice counts
- **Focus on how you FEEL** - energy and mood improvements often come before physical changes
- **Be flexible** - swap similar foods or adjust timing as needed
- **Celebrate small wins** - completing each day is an achievement!

Your program is designed to work with your lifestyle while addressing all three of your goals: building exercise consistency, improving your vegetarian diet, and managing work stress for better sleep. The specialists and I have created something truly comprehensive that should give you noticeable improvements in energy and overall wellbeing!

Ready to start your wellness transformation? Let me know if you have any questions about any part of your program! 🚀
# Review what was created
print("=" * 60)
print("FINAL TODO STATUS")
print("=" * 60)
print(list_todos.invoke({}))

print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)
for f in sorted(WORKSPACE.iterdir()):
    if f.is_file():
        print(f"  [FILE] {f.name} ({f.stat().st_size} bytes)")
    elif f.is_dir():
        print(f"  [DIR] {f.name}/")
============================================================
FINAL TODO STATUS
============================================================
✅ [todo_1] Check user profile and save vegetarian preference (completed)
✅ [todo_2] Create exercise plan (completed)
✅ [todo_3] Create nutrition plan (completed)
✅ [todo_4] Create stress/sleep management plan (completed)
✅ [todo_5] Integrate and save comprehensive program (completed)

============================================================
GENERATED FILES
============================================================
  [FILE] alex_2week_exercise_plan.md (3841 bytes)
  [FILE] alex_2week_integrated_wellness_program.md (9517 bytes)
  [FILE] alex_2week_nutrition_plan.md (6267 bytes)
  [FILE] alex_2week_stress_sleep_plan.md (8670 bytes)
  [FILE] alex_stress_management_sleep_program.txt (3976 bytes)
  [FILE] energy_boosting_vegetarian_foods.txt (1130 bytes)
  [DIR] exercise_program/
  [FILE] meal_plan_for_alex.txt (2517 bytes)
  [FILE] meal_prep_instructions_for_alex.txt (1679 bytes)
  [FILE] meditation_health_benefits_report.md (4280 bytes)
  [FILE] morning-energy-routine-guide.md (24041 bytes)
  [FILE] morning_routine_guide.md (8698 bytes)
  [FILE] personalized_sleep_improvement_plan.md (5989 bytes)
  [DIR] reports/
  [DIR] research/
  [FILE] sleep_research_summary.md (9833 bytes)
  [FILE] weekly_workout_plan.md (3239 bytes)
  [DIR] workspace/
# Read one of the generated files
files = list(WORKSPACE.glob("*.md"))
if files:
    print(f"\nContents of {files[0].name}:")
    print("=" * 60)
    print(files[0].read_text()[:2000] + "..." if len(files[0].read_text()) > 2000 else files[0].read_text())
Contents of meditation_health_benefits_report.md:
============================================================
# Health Benefits of Meditation — Evidence Summary

## Scope
Most clinical research on “meditation” evaluates **mindfulness-based interventions** such as **Mindfulness-Based Stress Reduction (MBSR)** and **Mindfulness-Based Cognitive Therapy (MBCT)**, usually delivered over ~8 weeks with instructor guidance and home practice. Evidence does not always generalize to all meditation styles.

Primary synthesis source:
- **NCCIH (NIH)** — *Meditation and Mindfulness: Effectiveness and Safety* https://www.nccih.nih.gov/health/meditation-and-mindfulness-effectiveness-and-safety

---

## Best-supported health benefits (strongest/most consistent outcomes)

### 1) Stress and psychological distress
Meditation—especially mindfulness programs—shows **moderate evidence** for improving stress-related outcomes and general psychological distress versus no treatment and some nonspecific controls.
- Source: NCCIH (NIH) https://www.nccih.nih.gov/health/meditation-and-mindfulness-effectiveness-and-safety

### 2) Anxiety symptoms
Across many randomized trials and meta-analyses, mindfulness meditation is associated with **reductions in anxiety symptoms**, typically **small-to-moderate** on average depending on population and comparison group.
- Source: NCCIH (NIH) https://www.nccih.nih.gov/health/meditation-and-mindfulness-effectiveness-and-safety

### 3) Depressive symptoms
Mindfulness-based interventions show **moderate evidence** for reducing depressive symptoms in some populations. MBCT is often studied for relapse prevention in recurrent depression, but results depend on patient selection and study design.
- Source: NCCIH (NIH) https://www.nccih.nih.gov/health/meditation-and-mindfulness-effectiveness-and-safety

### 4) Sleep quality / insomnia symptoms
Mindfulness meditation can **improve sleep quality** and **reduce insomnia symptoms** in some studies, with evidence described by NCCIH as supportive but varying by comparator (e.g., education control vs. established insomnia therapies)...
❓ Question #3:
What are the key considerations when designing subagent configurations?

Consider:

When should subagents share tools vs have distinct tools?
How do you decide which model to use for each subagent?
What's the right granularity for subagent specialization?
Answer:
When designing subagent configurations, the goal is to get the benefits of delegation (quality, speed, context isolation, cost control) without creating coordination chaos. The key is deciding tools, models, and specialization boundaries in a way that keeps the system predictable.

Subagents should share tools when they need to collaborate on the same artifacts or operate in the same “workspace” (for example, both research and writing agents reading/writing the same report files). Shared tools also make sense for common utilities like file read/write and basic formatting. Subagents should have distinct tools when you want strong separation of concerns or safety boundaries—e.g., only a “data-agent” can access sensitive sources, only a “web-agent” can browse, only a “writer-agent” can publish final outputs. Distinct tools are also useful when you want to reduce tool confusion and keep each agent’s action space small.

Model choice should follow a simple cost/quality rule: use the cheapest model that reliably does the job for that subtask, and reserve stronger models for tasks where quality matters most. For example:

Use cheaper models for lookup, summarization, extraction, drafting, simple transformations

Use stronger models for final synthesis, nuanced reasoning, safety-sensitive outputs, user-facing language Also consider latency: high-parallelism workflows benefit from faster models for most subagents.

Granularity is about avoiding two extremes: too broad (subagents become “mini general agents”) or too narrow (too many handoffs). A good specialization boundary is usually by function (research vs writing vs analysis) or by domain (exercise vs nutrition vs mindfulness). Each subagent should own a coherent skill set with clear inputs/outputs, so the main agent can delegate with a tight prompt and get back a usable summary instead of a whole conversation.

Summary: - Share tools when agents need shared artifacts; separate tools when you need guardrails or reduced action space. - Pick models based on task difficulty + risk + user visibility (cheap for substeps, strong for final). - Specialize subagents around stable boundaries (function/domain), and keep the number of subagents small enough to avoid coordination overhead.

❓ Question #4:
For a production wellness application using Deep Agents, what would you need to add?

Consider:

Safety guardrails for health advice
Persistent storage (not in-memory)
Multi-user support and isolation
Monitoring and observability
Cost management with subagents
Answer:
For a production wellness app, you would add strong safety, persistence, isolation, and operational controls around the Deep Agent core. The notebook version is a great prototype, but production needs guardrails and infrastructure.

i. Safety guardrails (health advice): You need policies that prevent harmful or inappropriate guidance. That includes clear disclaimers, “red flag” detection (e.g., chest pain, suicidal ideation, severe symptoms) with escalation guidance, contraindication checks using the user’s meds/allergies, and strict limits on giving diagnosis or medication instructions. Add validation layers for outputs (rule-based + model-based safety checks) and require citations/grounding when advice is sourced from documents.

ii. Persistent storage: Replace in-memory todo + files with real storage: - Todos/tasks → database tables (Postgres) with status history - Files/artifacts → object storage (S3/GCS) with versioning - Long-term memory → LangGraph Store backed by Postgres/Redis (or another durable store) - You also want retention and deletion policies for user data.

iii. Multi-user support + isolation: Each user needs isolation across: - Workspace/files (/workspace/<user_id>/...) - Memory namespaces ((user_id, "profile"), (user_id, "metrics"), etc.) - Access control (authN/authZ), encrypted storage, and audit logs

This prevents cross-user leakage and supports compliance.

iv. Monitoring and observability: Add tracing and metrics for: - Tool calls, latency, failures, retries - Token usage and spend per user/session - Safety events (blocked outputs, escalations) - Quality signals (user feedback, completion rate) - Use LangSmith/OpenTelemetry + centralized logs, plus dashboards and alerts.

v. Cost management with subagents: Use routing and budgets: - Cheaper models for extraction/summarization; stronger model for final synthesis - Hard limits: max tool calls, max tokens, max subagent spawns - Caching: reuse retrieved snippets, summaries, and generated artifacts - Batch/async for non-urgent tasks, and graceful degradation when budgets are hit

Summary : Production Deep Agents need a safety layer, durable storage, strict user isolation, strong observability, and cost controls so the system is trustworthy, scalable, and affordable.

🏗️ Activity #2: Build a Wellness Coach Agent
Build your own wellness coach that uses all 4 Deep Agent elements.

Requirements:
Planning: Create todos for a 30-day wellness challenge
Context Management: Store daily check-in notes
Subagents: At least 2 specialized subagents
Memory: Remember user preferences across interactions
Challenge:
Create a "30-Day Wellness Challenge" system that:

Generates a personalized 30-day plan
Tracks daily progress
Adapts recommendations based on feedback
Saves a weekly summary report
Step 0: Workspace + Backend

from pathlib import Path
from typing import List, Literal, Optional
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.store.memory import InMemoryStore

# -----------------------------
# Step 0: Workspace + Backend
# -----------------------------
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

filesystem_backend = FilesystemBackend(
    root_dir=str(WORKSPACE.absolute()),
    virtual_mode=True  # IMPORTANT sandbox
)
Step 1: Planning tools (todos)

# -----------------------------
# Step 1: Planning tools (todos)
# -----------------------------
TODO_STORE = {}

@tool
def write_todos(todos: List[dict]) -> str:
    """Create todos. Each item: {'title': str, 'description': str (optional)}"""
    created = []
    for t in todos:
        tid = f"todo_{len(TODO_STORE) + 1}"
        TODO_STORE[tid] = {"id": tid, "title": t.get("title","Untitled"),
                          "description": t.get("description",""), "status": "pending"}
        created.append(tid)
    return f"Created {len(created)} todos: {', '.join(created)}"

@tool
def update_todo(todo_id: str, status: Literal["pending", "in_progress", "completed"]) -> str:
    """Update todo status."""
    if todo_id not in TODO_STORE:
        return f"Todo not found: {todo_id}"
    TODO_STORE[todo_id]["status"] = status
    return f"Updated {todo_id} -> {status}"

@tool
def list_todos() -> str:
    """List todos."""
    if not TODO_STORE:
        return "No todos found"
    icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    return "\n".join(
        f"{icon.get(t['status'],'❓')} [{tid}] {t['title']} ({t['status']})"
        for tid, t in TODO_STORE.items()
    )
Step 2: Long-term memory tools

# -----------------------------
# Step 2: Long-term memory tools
# -----------------------------
memory_store = InMemoryStore()

@tool
def get_user_profile(user_id: str) -> str:
    """Get user profile/preferences from long-term memory."""
    ns_profile = (user_id, "profile")
    ns_prefs = (user_id, "preferences")
    prof = list(memory_store.search(ns_profile))
    prefs = list(memory_store.search(ns_prefs))

    if not prof and not prefs:
        return f"No profile/preferences found for {user_id}"

    lines = [f"User: {user_id}"]
    if prof:
        lines.append("Profile:")
        for item in prof:
            lines.append(f"  {item.key}: {item.value}")
    if prefs:
        lines.append("Preferences:")
        for item in prefs:
            lines.append(f"  {item.key}: {item.value}")
    return "\n".join(lines)

@tool
def save_user_preference(user_id: str, key: str, value: str) -> str:
    """Save a user preference (e.g., vegetarian, morning workouts, injuries)."""
    ns = (user_id, "preferences")
    memory_store.put(ns, key, {"value": value})
    return f"Saved preference for {user_id}: {key}={value}"
Step 3: Context mgmt tools for check-ins + weekly summaries (FILES) (Agent also has built-in write_file/read_file/ls/edit_file via backend.)

# -----------------------------
# Step 3: Context mgmt tools for check-ins + weekly summaries (FILES)
# (Agent also has built-in write_file/read_file/ls/edit_file via backend.)
# -----------------------------
@tool
def record_daily_checkin(user_id: str, day: int, checkin_text: str) -> str:
    """
    Store daily check-in notes to workspace/users/<user_id>/checkins/day_XX.md
    """
    base = WORKSPACE / "users" / user_id / "checkins"
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"day_{day:02d}.md"
    p.write_text(checkin_text, encoding="utf-8")
    return f"Saved check-in: users/{user_id}/checkins/{p.name}"

@tool
def list_checkins(user_id: str) -> str:
    """List saved check-ins for a user."""
    base = WORKSPACE / "users" / user_id / "checkins"
    if not base.exists():
        return "(no check-ins yet)"
    files = sorted([f.name for f in base.glob("day_*.md")])
    return "\n".join(files) if files else "(no check-ins yet)"

@tool
def save_weekly_summary(user_id: str, week: int, summary_md: str) -> str:
    """
    Save weekly summary to workspace/users/<user_id>/summaries/week_XX.md
    """
    base = WORKSPACE / "users" / user_id / "summaries"
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"week_{week:02d}.md"
    p.write_text(summary_md, encoding="utf-8")
    return f"Saved weekly summary: users/{user_id}/summaries/{p.name}"
Step 4: Subagent configurations (at least 2)

# -----------------------------
# Step 4: Subagent configurations (at least 2)
# -----------------------------
exercise_specialist = {
    "name": "exercise-specialist",
    "description": "Creates safe, progressive workout plans and adaptations.",
    "system_prompt": """You are an exercise specialist.
Provide clear workouts, progressions, and modifications.
Ask for constraints if needed (time, injuries, equipment).""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}

nutrition_specialist = {
    "name": "nutrition-specialist",
    "description": "Creates meal guidance/plans respecting dietary preferences and constraints.",
    "system_prompt": """You are a nutrition specialist.
Respect diet preferences (e.g., vegetarian), focus on practical meals and habits.
Avoid medical claims; recommend professional advice when needed.""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}

mindfulness_specialist = {
    "name": "mindfulness-specialist",
    "description": "Stress, sleep, mindfulness routines, and coping strategies.",
    "system_prompt": """You are a mindfulness specialist.
Give practical stress/sleep techniques and habit plans.
Flag red-flag symptoms and recommend professional help when appropriate.""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}
Step 5: Main coordinator agent (all 4 elements)

# -----------------------------
# Step 5: Main coordinator agent (all 4 elements)
# -----------------------------
TODO_STORE.clear()

wellness_coach_30d = create_deep_agent(
    model=init_chat_model("openai:gpt-4o-mini"),
    #model=init_chat_model("openai:gpt-4o"),  # coordinator slightly stronger; can use 4o-mini too
    tools=[
        # planning
        write_todos, update_todo, list_todos,
        # memory
        get_user_profile, save_user_preference,
        # context mgmt (files)
        record_daily_checkin, list_checkins, save_weekly_summary,
    ],
    backend=filesystem_backend,
    subagents=[exercise_specialist, nutrition_specialist, mindfulness_specialist],
    system_prompt="""
You are a 30-Day Wellness Challenge Coach.

Planning is REQUIRED.

Before producing ANY plan:
- You MUST call write_todos() with 6–10 todos for the 30-day challenge.
- Immediately set the first todo to in_progress.
- As you complete steps, update todos to completed.

Workflow you must follow:

1. Get user profile and preferences.
2. Create todos for:
   - Assess goals
   - Create exercise plan
   - Create nutrition plan
   - Create mindfulness plan
   - Integrate into 30-day program
   - Save plan to file
3. Use subagents for exercise, nutrition, and mindfulness content.
4. Save the final plan to:
   users/<user_id>/plans/30_day_plan.md
5. At the end, show the todo list using list_todos.

If todos are not created, STOP and create them first.
"""
)
Step 6: TEST — (A) Create 30-day challenge plan

# ============================================================
# Step 6: TEST — (A) Create 30-day challenge plan
# ============================================================
user_id = "user_alex"  # change if you want

# store a couple of preferences so memory is demonstrated
save_user_preference.invoke({"user_id": user_id, "key": "diet", "value": "vegetarian"})
save_user_preference.invoke({"user_id": user_id, "key": "workout_time", "value": "morning"})
save_user_preference.invoke({"user_id": user_id, "key": "available_workouts", "value": "3x/week, 30 mins"})

create_plan_prompt = f"""
My user_id is {user_id}.

IMPORTANT:
First create a todo list for the 30-day challenge using write_todos.

Then:
- Build the full 30-day wellness program
- Save it to users/{user_id}/plans/30_day_plan.md
- Finally show me the todo list using list_todos
"""


result1 = wellness_coach_30d.invoke({"messages": [{"role": "user", "content": create_plan_prompt}]})
print(result1["messages"][-1].content)

print("\n--- TODOS AFTER PLAN ---")
print(list_todos.invoke({}))
The 30-Day Wellness Challenge program has been successfully created and saved. Here’s a summary of the todos:

1. **Assess goals** - *in progress*
2. **Create exercise plan** - *completed*
3. **Create nutrition plan** - *completed*
4. **Create mindfulness plan** - *completed*
5. **Integrate into 30-day program** - *completed*
6. **Save plan to file** - *completed*

If you have any further questions or need assistance, feel free to ask!

--- TODOS AFTER PLAN ---
🔄 [todo_1] Assess goals (in_progress)
✅ [todo_2] Create exercise plan (completed)
✅ [todo_3] Create nutrition plan (completed)
✅ [todo_4] Create mindfulness plan (completed)
✅ [todo_5] Integrate into 30-day program (completed)
✅ [todo_6] Save plan to file (completed)
Step 7: TEST — (B) Simulate a daily check-in + adaptation

# ============================================================
# Step 7: TEST — (B) Simulate a daily check-in + adaptation
# ============================================================
daily_checkin_prompt = f"""
My user_id is {user_id}.

Day 1 check-in:
- Slept 5.5 hours, woke up twice
- Stress 8/10, lots of meetings
- Did not exercise
- Had late caffeine at 4pm

REQUIREMENTS:
1) Save this check-in using record_daily_checkin(user_id, day=1, ...).
2) Adapt tomorrow (Day 2) recommendations based on this feedback.
3) Update relevant todos (use update_todo) and then show the todo list (use list_todos).
"""


result2 = wellness_coach_30d.invoke({"messages": [{"role": "user", "content": daily_checkin_prompt}]})
print("\n" + result2["messages"][-1].content)

print("\n--- CHECKINS LIST ---")
print(list_checkins.invoke({"user_id": user_id}))
The check-in for Day 1 has been successfully saved. Here's a summary of your check-in:
- **Sleep**: 5.5 hours, woke up twice
- **Stress Level**: 8/10 (lots of meetings)
- **Exercise**: Did not exercise
- **Caffeine Intake**: Late caffeine at 4 pm

### 1. Adapted Recommendations for Tomorrow (Day 2):
- **Sleep Improvement**: Aim for earlier bedtime to increase sleep duration. Avoid caffeine after noon.
- **Stress Management**: Incorporate mindfulness exercises or short breaks between meetings to reduce stress levels.
- **Exercise**: Aim for a brief walk or light workout to help alleviate stress and improve mood.

### 2. Todo List:
- [🔄 Assess goals (in_progress)]
- [✅ Create exercise plan (completed)]
- [✅ Create nutrition plan (completed)]
- [✅ Create mindfulness plan (completed)]
- [✅ Integrate into 30-day program (completed)]
- [✅ Save plan to file (completed)]

Let me know if you need any further information or adjustments!

--- CHECKINS LIST ---
day_01.md
Step 8: TEST — (C) Simulate end-of-week summary (week 1) - (You can call this after several check-ins; here we simulate directly.)

# ============================================================
# Step 8: TEST — (C) Simulate end-of-week summary (week 1)
# (You can call this after several check-ins; here we simulate directly.)
# ============================================================
weekly_summary_prompt = f"""
My user_id is {user_id}.

REQUIREMENTS:
1) Use list_checkins(user_id) to see what check-ins exist.
2) Create a Week 1 summary (wins, misses, patterns, Week 2 adjustments).
3) Save it using save_weekly_summary(user_id, week=1, ...).
4) Update relevant todos (use update_todo) and then show the todo list (use list_todos).
"""


result3 = wellness_coach_30d.invoke({"messages": [{"role": "user", "content": weekly_summary_prompt}]})
print("\n" + result3["messages"][-1].content)

print("\n--- DONE ---")
Here's the summary of your current activities:

### Week 1 Summary
- **Wins**: Engaged in daily check-ins to track wellness journey.
- **Misses**: No significant entries to evaluate due to lack of check-in data.
- **Patterns**: Consistent efforts to maintain wellness goals, but specific patterns cannot be established.
- **Week 2 Adjustments**: Encouragement to engage more actively in daily check-ins. Focus on specific goals such as nutrition and exercise for clearer evaluations.

### To-Do List Status
- ✅ Assess goals (completed)
- ✅ Create exercise plan (completed)
- ✅ Create nutrition plan (completed)
- ✅ Create mindfulness plan (completed)
- ✅ Integrate into 30-day program (completed)
- ✅ Save plan to file (completed)

All tasks have been successfully completed. If you have any further tasks or adjustments, just let me know!

--- DONE ---
Advanced Build : Build the AI Life Coach with Multi-Domain Expertise as described in the README.md Advanced Build section
# ============================================================
# Advanced Build :  AI Life Coach with Multi-Domain Expertise
# Deep Agents: Planning + Context Mgmt + Subagents + Long-term Memory
# ============================================================

from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.store.memory import InMemoryStore
import json
import datetime

# -----------------------------
# 0) Workspace + Backend
# -----------------------------
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

filesystem_backend = FilesystemBackend(
    root_dir=str(WORKSPACE.absolute()),
    virtual_mode=True  # REQUIRED sandbox
)

# -----------------------------
# 1) Advanced Planning tools (with phases + dependencies)
# -----------------------------
PLAN_STORE: Dict[str, Any] = {}  # in-memory demo; production -> DB

@tool
def write_life_todos(user_id: str, todos: List[dict]) -> str:
    """
    Create multi-phase planning todos with optional dependencies.

    Each todo dict can include:
      - title (str)
      - phase (discovery/planning/execution)
      - depends_on (str optional: title of dependency)
    """
    PLAN_STORE.setdefault(user_id, {})
    PLAN_STORE[user_id].setdefault("todos", [])
    # normalize and create IDs
    created = []
    for i, t in enumerate(todos, start=1):
        tid = f"{user_id}_todo_{len(PLAN_STORE[user_id]['todos']) + 1}"
        item = {
            "id": tid,
            "title": t.get("title", "Untitled"),
            "phase": t.get("phase", "execution"),
            "depends_on": t.get("depends_on"),
            "status": "pending"
        }
        PLAN_STORE[user_id]["todos"].append(item)
        created.append(tid)
    return f"Created {len(created)} todos for {user_id}: {', '.join(created)}"

@tool
def update_life_todo(user_id: str, todo_id: str, status: Literal["pending","in_progress","completed"]) -> str:
    """Update a planning todo status."""
    todos = PLAN_STORE.get(user_id, {}).get("todos", [])
    for t in todos:
        if t["id"] == todo_id:
            t["status"] = status
            return f"Updated {todo_id} -> {status}"
    return f"Todo not found: {todo_id}"

@tool
def list_life_todos(user_id: str) -> str:
    """List the user's planning todos (with phase + dependency + status)."""
    todos = PLAN_STORE.get(user_id, {}).get("todos", [])
    if not todos:
        return "No todos found"
    icon = {"pending":"⬜", "in_progress":"🔄", "completed":"✅"}
    lines = []
    for t in todos:
        dep = f" | depends_on={t['depends_on']}" if t.get("depends_on") else ""
        lines.append(f"{icon.get(t['status'],'❓')} {t['id']} | {t['phase']} | {t['title']}{dep}")
    return "\n".join(lines)

# -----------------------------
# 2) Context Management Strategy (folder layout)
#    user_profile/, assessments/, plans/, progress/, resources/
# -----------------------------
def _user_root(user_id: str) -> Path:
    return WORKSPACE / "users" / user_id

@tool
def ensure_user_folders(user_id: str) -> str:
    """Create required folder structure for the life coach."""
    root = _user_root(user_id)
    for folder in ["user_profile", "assessments", "plans", "progress", "resources"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    return f"Ensured folders under: users/{user_id}/ (user_profile, assessments, plans, progress, resources)"

@tool
def save_assessment(user_id: str, assessment_md: str) -> str:
    """Save an assessment markdown to assessments/ with timestamp."""
    root = _user_root(user_id) / "assessments"
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"assessment_{ts}.md"
    path.write_text(assessment_md, encoding="utf-8")
    return f"Saved assessment: users/{user_id}/assessments/{path.name}"

@tool
def save_action_plan(user_id: str, plan_md: str, label: str = "90_day_action_plan") -> str:
    """Save the active plan to plans/ (overwrite the active file for simplicity)."""
    root = _user_root(user_id) / "plans"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}.md"
    path.write_text(plan_md, encoding="utf-8")
    return f"Saved plan: users/{user_id}/plans/{path.name}"

@tool
def record_weekly_checkin(user_id: str, week: int, checkin_md: str) -> str:
    """Save weekly check-in notes to progress/week_XX_checkin.md"""
    root = _user_root(user_id) / "progress"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"week_{week:02d}_checkin.md"
    path.write_text(checkin_md, encoding="utf-8")
    return f"Saved check-in: users/{user_id}/progress/{path.name}"

@tool
def save_weekly_summary(user_id: str, week: int, summary_md: str) -> str:
    """Save weekly summary to progress/week_XX_summary.md"""
    root = _user_root(user_id) / "progress"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"week_{week:02d}_summary.md"
    path.write_text(summary_md, encoding="utf-8")
    return f"Saved weekly summary: users/{user_id}/progress/{path.name}"

# -----------------------------
# 3) Memory Integration (namespaces per README)
# -----------------------------
memory_store = InMemoryStore()

@tool
def mem_get(user_id: str, bucket: str) -> str:
    """Get all items from a memory namespace: (user_id, bucket)"""
    ns = (user_id, bucket)
    items = list(memory_store.search(ns))
    if not items:
        return f"(empty) namespace={ns}"
    lines = [f"namespace={ns}"]
    for it in items:
        lines.append(f"- {it.key}: {it.value}")
    return "\n".join(lines)

@tool
def mem_put(user_id: str, bucket: str, key: str, value_json: str) -> str:
    """
    Put JSON value into memory namespace.
    Example: mem_put("user_alex","preferences","tone","{\"value\":\"concise\"}")
    """
    ns = (user_id, bucket)
    try:
        value = json.loads(value_json)
    except Exception:
        value = {"value": value_json}
    memory_store.put(ns, key, value)
    return f"Saved to namespace={(user_id,bucket)} key={key}"

@tool
def mem_put_shared(bucket: str, key: str, value_json: str) -> str:
    """Put anonymized cross-user pattern to shared namespace: ('coaching', bucket)"""
    ns = ("coaching", bucket)
    try:
        value = json.loads(value_json)
    except Exception:
        value = {"value": value_json}
    memory_store.put(ns, key, value)
    return f"Saved to shared namespace={ns} key={key}"

# -----------------------------
# 4) Subagent Configurations (Career, Relationship, Finance, Wellness)
# -----------------------------
career_coach = {
    "name": "career-coach",
    "description": "Job search, skill development, professional growth, strategy.",
    "system_prompt": """You are a Career Coach.
Deliver: clear options, next steps, and a realistic 2-week sprint plan.
Ask for role target, constraints, and timeline if missing.""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}

relationship_coach = {
    "name": "relationship-coach",
    "description": "Communication, boundaries, social connection, conflict navigation.",
    "system_prompt": """You are a Relationship Coach.
Deliver: scripts, boundary-setting steps, and reflection prompts.
Be practical and empathetic.""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}

finance_coach = {
    "name": "finance-coach",
    "description": "Budgeting, saving, debt payoff, financial goals.",
    "system_prompt": """You are a Finance Coach.
Deliver: a simple budget framework, prioritization, and weekly habits.
Avoid regulated financial advice; focus on general best practices.""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}

wellness_coach = {
    "name": "wellness-coach",
    "description": "Health, fitness, sleep, stress management.",
    "system_prompt": """You are a Wellness Coach.
Deliver: routines that are safe, sustainable, and habit-based.
Include guardrails: consult professionals for serious symptoms.""",
    "tools": [],
    "model": "openai:gpt-4o-mini",
}

# -----------------------------
# 5) Life Coach Coordinator (Deep Agent)
# -----------------------------
life_coach = create_deep_agent(
    model=init_chat_model("openai:gpt-4o-mini"),  # swap to Claude later if you want
    tools=[
        # planning
        write_life_todos, update_life_todo, list_life_todos,
        # context
        ensure_user_folders, save_assessment, save_action_plan,
        record_weekly_checkin, save_weekly_summary,
        # memory
        mem_get, mem_put, mem_put_shared,
    ],
    backend=filesystem_backend,
    subagents=[career_coach, relationship_coach, finance_coach, wellness_coach],
    system_prompt="""
You are the Life Coach Coordinator.

You MUST use all 4 Deep Agent elements:
1) Planning: create multi-phase todos with dependencies (write_life_todos).
2) Context Management: write artifacts into the required folder structure:
   users/<user_id>/{user_profile,assessments,plans,progress,resources}/
3) Subagents: delegate domain work to specialist coaches (career, relationship, finance, wellness).
4) Long-term Memory: store and reuse:
   (user_id,"profile"), (user_id,"goals"), (user_id,"progress"), (user_id,"preferences")
   and optionally ("coaching","patterns") anonymized insights.

Workflow:
A) Ensure user folders exist.
B) Pull existing memory (profile/goals/preferences/progress).
C) Run an initial life assessment (save_assessment).
D) Identify top 3 priorities.
E) Create a 90-day action plan with cross-domain integration (save_action_plan).
F) Set up a weekly check-in system (record_weekly_checkin template) and weekly summaries.
G) When user gives feedback/check-in: update plan + save weekly summary.

Output rules:
- Be concise.
- Always include the file paths you saved.
- Always show current todos at the end using list_life_todos(user_id).
"""
)

# -----------------------------
# 6) Architecture Diagram (Mermaid) - paste into QUICKSTART later
# -----------------------------
ARCH_DIAGRAM = r"""
```mermaid
flowchart TB
  U[User] --> C[Life Coach Coordinator]

  C -->|task delegation| CC[Career Coach]
  C -->|task delegation| RC[Relationship Coach]
  C -->|task delegation| FC[Finance Coach]
  C -->|task delegation| WC[Wellness Coach]

  C --> FS[(Workspace Filesystem)]
  C --> MS[(LangGraph Store Memory)]

  FS --> P[plans/]
  FS --> A[assessments/]
  FS --> PR[progress/]
  FS --> UP[user_profile/]
  FS --> R[resources/]
  """
  #print("Advanced Build Life Coach loaded. Mermaid diagram available in ARCH_DIAGRAM.")

### How to use it (quick test)
#Run this prompt to validate multi-domain orchestration:

#```python
user_id = "user_alex"

bootstrap = f"""
My user_id is {user_id}.
I feel stuck in my career and it's affecting my relationships. I'm also stressed about money and my sleep is getting worse.

Preferences:
- Coaching style: concise, action-oriented
- Schedule: busy weekdays, more time on weekends

Please:
1) Create multi-phase todos with dependencies
2) Save an initial assessment
3) Identify top 3 priorities
4) Create a 90-day integrated action plan and save it
5) Set up a weekly check-in template
"""
out = life_coach.invoke({"messages":[{"role":"user","content":bootstrap}]})
print(out["messages"][-1].content)
Here's what has been accomplished for you:

1. **Folders Created:** User folders set up under `users/user_alex/`.
2. **Initial Assessment Saved:** 
   - Path: `users/user_alex/assessments/assessment_20260209_214915.md`
3. **Multi-phase To-Do List Created:** Focus areas include Career, Relationships, Finance, and Wellness.
4. **90-Day Integrated Action Plan Created and Saved:** 
   - Path: `users/user_alex/plans/90_day_action_plan.md`
5. **Weekly Check-in Template Established:** 
   - Path: `users/user_alex/progress/week_01_checkin.md`

### Current To-Dos:
- **Career:** Assess current career satisfaction and goal-setting.
- **Relationship:** Evaluate dynamics and improve communication.
- **Finance:** Review financial situation and establish a budget.
- **Wellness:** Assess and improve health and sleep.

Feel free to reach out if you need further adjustments or specifics in the action plan!
Summary
In this session, we explored Deep Agents and their four key elements:

Element	Purpose	Implementation
Planning	Track complex tasks	write_todos, update_todo, list_todos
Context Management	Handle large contexts	File system tools, automatic offloading
Subagent Spawning	Delegate to specialists	task tool with custom configs
Long-term Memory	Remember across sessions	LangGraph Store integration
Key Takeaways:
Deep Agents handle complexity - Unlike simple tool loops, they can manage long-horizon, multi-step tasks
Planning is context engineering - Todo lists and files aren't just organization—they're extended memory
Subagents prevent context bloat - Delegation keeps the main agent focused and efficient
Skills enable progressive disclosure - Load capabilities on-demand instead of upfront
The CLI makes interaction natural - Interactive sessions with conversation resume
Deep Agents vs Traditional Agents
Aspect	Traditional Agent	Deep Agent
Task complexity	Simple, single-step	Complex, multi-step
Context management	All in conversation	Files + summaries
Delegation	None	Subagent spawning
Memory	Within thread	Across sessions
Planning	Implicit	Explicit (todos)
When to Use Deep Agents
Use Deep Agents when:

Tasks require multiple steps or phases
Context would overflow in a simple loop
Specialization would improve quality
Users need to resume sessions
Long-term memory is valuable
Use Simple Agents when:

Tasks are straightforward Q&A
Single tool call suffices
Context fits easily
No need for persistence
Further Reading
Deep Agents Documentation
Deep Agents GitHub
Context Management Blog Post
Building Multi-Agent Applications
LangGraph Memory Concepts