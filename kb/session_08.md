
LangGraph Open Deep Research - Supervisor-Researcher Architecture
In this notebook, we'll explore the supervisor-researcher delegation architecture for conducting deep research with LangGraph.

You can visit this repository to see the original application: Open Deep Research

Let's jump in!

What We're Building
This implementation uses a hierarchical delegation pattern where:

User Clarification - Optionally asks clarifying questions to understand the research scope
Research Brief Generation - Transforms user messages into a structured research brief
Supervisor - A lead researcher that analyzes the brief and delegates research tasks
Parallel Researchers - Multiple sub-agents that conduct focused research simultaneously
Research Compression - Each researcher synthesizes their findings
Final Report - All findings are combined into a comprehensive report
Architecture Diagram

This differs from a section-based approach by allowing dynamic task decomposition based on the research question, rather than predefined sections.

🤝 Breakout Room #1
Deep Research Foundations
In this breakout room, we'll understand the architecture and components of the Open Deep Research system.

Task 1: Dependencies
You'll need API keys for Anthropic (for the LLM) and Tavily (for web search). We'll configure the system to use Anthropic's Claude Sonnet 4 exclusively.

import os
import getpass

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")
Task 2: State Definitions
The state structure is hierarchical with three levels:

Agent State (Top Level)
Contains the overall conversation messages, research brief, accumulated notes, and final report.

Supervisor State (Middle Level)
Manages the research supervisor's messages, research iterations, and coordinating parallel researchers.

Researcher State (Bottom Level)
Each individual researcher has their own message history, tool call iterations, and research findings.

We also have structured outputs for tool calling:

ConductResearch - Tool for supervisor to delegate research to a sub-agent
ResearchComplete - Tool to signal research phase is done
ClarifyWithUser - Structured output for asking clarifying questions
ResearchQuestion - Structured output for the research brief
Let's import these from our library: open_deep_library/state.py

# Import state definitions from the library
from open_deep_library.state import (
    # Main workflow states
    AgentState,           # Lines 65-72: Top-level agent state with messages, research_brief, notes, final_report
    AgentInputState,      # Lines 62-63: Input state is just messages
    
    # Supervisor states
    SupervisorState,      # Lines 74-81: Supervisor manages research delegation and iterations
    
    # Researcher states
    ResearcherState,      # Lines 83-90: Individual researcher with messages and tool iterations
    ResearcherOutputState, # Lines 92-96: Output from researcher (compressed research + raw notes)
    
    # Structured outputs for tool calling
    ConductResearch,      # Lines 15-19: Tool for delegating research to sub-agents
    ResearchComplete,     # Lines 21-22: Tool to signal research completion
    ClarifyWithUser,      # Lines 30-41: Structured output for user clarification
    ResearchQuestion,     # Lines 43-48: Structured output for research brief
)
Task 3: Utility Functions and Tools
The system uses several key utilities:

Search Tools
tavily_search - Async web search with automatic summarization to stay within token limits
Supports Anthropic native web search and Tavily API
Reflection Tools
think_tool - Allows researchers to reflect on their progress and plan next steps (ReAct pattern)
Helper Utilities
get_all_tools - Assembles the complete toolkit (search + MCP + reflection)
get_today_str - Provides current date context for research
Token limit handling utilities for graceful degradation
These are defined in open_deep_library/utils.py

import os
print("repo listing:", os.listdir())
print("has open_deep_research dir:", os.path.isdir("open_deep_research"))
repo listing: ['open_deep_research.egg-info', 'open_deep_library', 'uv.lock', 'pyproject.toml', 'README.md', '.venv', 'open-deep-research.ipynb', 'data']
has open_deep_research dir: False
# Import utility functions and tools from the library
from open_deep_library.utils import (
    # Search tool - Lines 43-136: Tavily search with automatic summarization
    tavily_search,
    
    # Reflection tool - Lines 219-244: Strategic thinking tool for ReAct pattern
    think_tool,
    
    # Tool assembly - Lines 569-597: Get all configured tools
    get_all_tools,
    
    # Date utility - Lines 872-879: Get formatted current date
    get_today_str,
    
    # Supporting utilities for error handling
    get_api_key_for_model,          # Lines 892-914: Get API keys from config or env
    is_token_limit_exceeded,         # Lines 665-701: Detect token limit errors
    get_model_token_limit,           # Lines 831-846: Look up model's token limit
    remove_up_to_last_ai_message,    # Lines 848-866: Truncate messages for retry
    anthropic_websearch_called,      # Lines 607-637: Detect Anthropic native search usage
    openai_websearch_called,         # Lines 639-658: Detect OpenAI native search usage
    get_notes_from_tool_calls,       # Lines 599-601: Extract notes from tool messages
)
---------------------------------------------------------------------------

ModuleNotFoundError                       Traceback (most recent call last)

Cell In[1], line 2

      1 # Import utility functions and tools from the library

----> 2 from open_deep_library.utils import (

      3     # Search tool - Lines 43-136: Tavily search with automatic summarization

      4     tavily_search,

      5 

      6     # Reflection tool - Lines 219-244: Strategic thinking tool for ReAct pattern

      7     think_tool,

      8 

      9     # Tool assembly - Lines 569-597: Get all configured tools

     10     get_all_tools,

     11 

     12     # Date utility - Lines 872-879: Get formatted current date

     13     get_today_str,

     14 

     15     # Supporting utilities for error handling

     16     get_api_key_for_model,          # Lines 892-914: Get API keys from config or env

     17     is_token_limit_exceeded,         # Lines 665-701: Detect token limit errors

     18     get_model_token_limit,           # Lines 831-846: Look up model's token limit

     19     remove_up_to_last_ai_message,    # Lines 848-866: Truncate messages for retry

     20     anthropic_websearch_called,      # Lines 607-637: Detect Anthropic native search usage

     21     openai_websearch_called,         # Lines 639-658: Detect OpenAI native search usage

     22     get_notes_from_tool_calls,       # Lines 599-601: Extract notes from tool messages

     23 )



File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/utils.py:32

     29 from mcp import McpError

     30 from tavily import AsyncTavilyClient

---> 32 from open_deep_research.configuration import Configuration, SearchAPI

     33 from open_deep_research.prompts import summarize_webpage_prompt

     34 from open_deep_research.state import ResearchComplete, Summary



ModuleNotFoundError: No module named 'open_deep_research'
# Import utility functions and tools from the library
from open_deep_library.utils import (
    # Search tool - Lines 43-136: Tavily search with automatic summarization
    tavily_search,
    
    # Reflection tool - Lines 219-244: Strategic thinking tool for ReAct pattern
    think_tool,
    
    # Tool assembly - Lines 569-597: Get all configured tools
    get_all_tools,
    
    # Date utility - Lines 872-879: Get formatted current date
    get_today_str,
    
    # Supporting utilities for error handling
    get_api_key_for_model,          # Lines 892-914: Get API keys from config or env
    is_token_limit_exceeded,         # Lines 665-701: Detect token limit errors
    get_model_token_limit,           # Lines 831-846: Look up model's token limit
    remove_up_to_last_ai_message,    # Lines 848-866: Truncate messages for retry
    anthropic_websearch_called,      # Lines 607-637: Detect Anthropic native search usage
    openai_websearch_called,         # Lines 639-658: Detect OpenAI native search usage
    get_notes_from_tool_calls,       # Lines 599-601: Extract notes from tool messages
)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[4], line 2
      1 # Import utility functions and tools from the library
----> 2 from open_deep_library.utils import (
      3     # Search tool - Lines 43-136: Tavily search with automatic summarization
      4     tavily_search,
      5 
      6     # Reflection tool - Lines 219-244: Strategic thinking tool for ReAct pattern
      7     think_tool,
      8 
      9     # Tool assembly - Lines 569-597: Get all configured tools
     10     get_all_tools,
     11 
     12     # Date utility - Lines 872-879: Get formatted current date
     13     get_today_str,
     14 
     15     # Supporting utilities for error handling
     16     get_api_key_for_model,          # Lines 892-914: Get API keys from config or env
     17     is_token_limit_exceeded,         # Lines 665-701: Detect token limit errors
     18     get_model_token_limit,           # Lines 831-846: Look up model's token limit
     19     remove_up_to_last_ai_message,    # Lines 848-866: Truncate messages for retry
     20     anthropic_websearch_called,      # Lines 607-637: Detect Anthropic native search usage
     21     openai_websearch_called,         # Lines 639-658: Detect OpenAI native search usage
     22     get_notes_from_tool_calls,       # Lines 599-601: Extract notes from tool messages
     23 )

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/utils.py:32
     29 from mcp import McpError
     30 from tavily import AsyncTavilyClient
---> 32 from open_deep_research.configuration import Configuration, SearchAPI
     33 from open_deep_research.prompts import summarize_webpage_prompt
     34 from open_deep_research.state import ResearchComplete, Summary

ModuleNotFoundError: No module named 'open_deep_research'
# Import utility functions and tools from the library
from open_deep_library.utils import (
    # Search tool - Lines 43-136: Tavily search with automatic summarization
    tavily_search,
    
    # Reflection tool - Lines 219-244: Strategic thinking tool for ReAct pattern
    think_tool,
    
    # Tool assembly - Lines 569-597: Get all configured tools
    get_all_tools,
    
    # Date utility - Lines 872-879: Get formatted current date
    get_today_str,
    
    # Supporting utilities for error handling
    get_api_key_for_model,          # Lines 892-914: Get API keys from config or env
    is_token_limit_exceeded,         # Lines 665-701: Detect token limit errors
    get_model_token_limit,           # Lines 831-846: Look up model's token limit
    remove_up_to_last_ai_message,    # Lines 848-866: Truncate messages for retry
    anthropic_websearch_called,      # Lines 607-637: Detect Anthropic native search usage
    openai_websearch_called,         # Lines 639-658: Detect OpenAI native search usage
    get_notes_from_tool_calls,       # Lines 599-601: Extract notes from tool messages
)
---------------------------------------------------------------------------

ModuleNotFoundError                       Traceback (most recent call last)

Cell In[1], line 2

      1 # Import utility functions and tools from the library

----> 2 from open_deep_library.utils import (

      3     # Search tool - Lines 43-136: Tavily search with automatic summarization

      4     tavily_search,

      5 

      6     # Reflection tool - Lines 219-244: Strategic thinking tool for ReAct pattern

      7     think_tool,

      8 

      9     # Tool assembly - Lines 569-597: Get all configured tools

     10     get_all_tools,

     11 

     12     # Date utility - Lines 872-879: Get formatted current date

     13     get_today_str,

     14 

     15     # Supporting utilities for error handling

     16     get_api_key_for_model,          # Lines 892-914: Get API keys from config or env

     17     is_token_limit_exceeded,         # Lines 665-701: Detect token limit errors

     18     get_model_token_limit,           # Lines 831-846: Look up model's token limit

     19     remove_up_to_last_ai_message,    # Lines 848-866: Truncate messages for retry

     20     anthropic_websearch_called,      # Lines 607-637: Detect Anthropic native search usage

     21     openai_websearch_called,         # Lines 639-658: Detect OpenAI native search usage

     22     get_notes_from_tool_calls,       # Lines 599-601: Extract notes from tool messages

     23 )



File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/utils.py:32

     29 from mcp import McpError

     30 from tavily import AsyncTavilyClient

---> 32 from open_deep_research.configuration import Configuration, SearchAPI

     33 from open_deep_research.prompts import summarize_webpage_prompt

     34 from open_deep_research.state import ResearchComplete, Summary



ModuleNotFoundError: No module named 'open_deep_research'
Task 4: Configuration System
The configuration system controls:

Research Behavior
allow_clarification - Whether to ask clarifying questions before research
max_concurrent_research_units - How many parallel researchers can run (default: 5)
max_researcher_iterations - How many times supervisor can delegate research (default: 6)
max_react_tool_calls - Tool call limit per researcher (default: 10)
Model Configuration
research_model - Model for research and supervision (we'll use Anthropic)
compression_model - Model for synthesizing findings
final_report_model - Model for writing the final report
summarization_model - Model for summarizing web search results
Search Configuration
search_api - Which search API to use (ANTHROPIC, TAVILY, or NONE)
max_content_length - Character limit before summarization
Defined in open_deep_library/configuration.py

# Import configuration from the library
from open_deep_library.configuration import (
    Configuration,    # Lines 38-247: Main configuration class with all settings
    SearchAPI,        # Lines 11-17: Enum for search API options (ANTHROPIC, TAVILY, NONE)
)
Task 5: Prompt Templates
The system uses carefully engineered prompts for each phase:

Phase 1: Clarification
clarify_with_user_instructions - Analyzes if the research scope is clear or needs clarification

Phase 2: Research Brief
transform_messages_into_research_topic_prompt - Converts user messages into a detailed research brief

Phase 3: Supervisor
lead_researcher_prompt - System prompt for the supervisor that manages delegation strategy

Phase 4: Researcher
research_system_prompt - System prompt for individual researchers conducting focused research

Phase 5: Compression
compress_research_system_prompt - Prompt for synthesizing research findings without losing information

Phase 6: Final Report
final_report_generation_prompt - Comprehensive prompt for writing the final report

All prompts are defined in open_deep_library/prompts.py

# Import prompt templates from the library
from open_deep_library.prompts import (
    clarify_with_user_instructions,                    # Lines 3-41: Ask clarifying questions
    transform_messages_into_research_topic_prompt,     # Lines 44-77: Generate research brief
    lead_researcher_prompt,                            # Lines 79-136: Supervisor system prompt
    research_system_prompt,                            # Lines 138-183: Researcher system prompt
    compress_research_system_prompt,                   # Lines 186-222: Research compression prompt
    final_report_generation_prompt,                    # Lines 228-308: Final report generation
)
❓ Question #1:
Explain the interrelationships between the three states (Agent, Supervisor, Researcher). Why don't we just make a single huge state?

Answer:
The AgentState is the top-level “case file” for the entire run: it holds the user conversation (messages), the derived research_brief, the accumulated notes coming back from all delegated work, and eventually the final_report. It’s the only state that spans the full lifecycle from clarification → brief → delegation → synthesis, so it acts as the stable interface between the user-facing flow and everything happening underneath.

The SupervisorState is a middle-layer control loop that lives inside the Agent’s run. It tracks the supervisor’s own message history (supervisor_messages) and iteration counters so the system can repeatedly decide “what to research next,” issue ConductResearch tool calls, and stop when ResearchComplete is emitted or limits are hit. This state is intentionally smaller and more operational than AgentState: it exists to manage delegation strategy and to coordinate multiple researcher executions without polluting the global conversation state with noisy internal steps.

The ResearcherState is the bottom-layer execution context for a single focused subtask. Each researcher maintains its own researcher_messages, tool-call iteration counters, and the raw findings produced by search + reflection. After the researcher finishes, it returns a ResearcherOutputState (compressed summary + raw notes) which is then appended/merged into the Supervisor/Agent-level notes. This “many small researcher states → one aggregated Agent notes pool” structure is what enables parallelism and avoids cross-talk.

We don’t make one huge state because it would collapse these clean boundaries and create practical failure modes: the context would balloon with every tool result and sub-agent trace, causing worse model performance (“attention budget” dilution), higher cost, and more token-limit crashes; parallel researchers would overwrite or interleave each other’s intermediate artifacts, making results nondeterministic and hard to debug; and you’d lose the ability to apply different policies per layer (e.g., strict tool limits for researchers, different prompts for supervisor vs researcher, targeted compression at the researcher boundary). The three-state hierarchy is basically scope control: it keeps global intent stable (Agent), delegation coherent (Supervisor), and execution isolated (Researcher), while allowing safe aggregation back upward.

❓ Question #2:
What are the advantages and disadvantages of importing these components instead of including them in the notebook?

Answer:
Importing components from open_deep_library turns the notebook into a thin orchestration layer rather than the place where the system lives. The biggest advantage is reusability and consistency: the same State schemas, prompts, node functions, and retry/token-limit logic can be used across multiple notebooks, scripts, and deployments without copy-paste drift. It also improves testability and maintainability because you can unit test deep_researcher.py nodes (e.g., token truncation, tool routing, stop conditions) outside the notebook, and a bug fix in one place updates every consumer. Practically, it also makes the notebook easier to read and teach from: the notebook shows the architecture and wiring (how graphs connect, how config is set, how streaming works) while the library holds the implementation details.

The trade-off is setup friction and debugging complexity. Notebooks are forgiving when everything is inline; imports require packaging correctness (module paths, editable installs, Python version constraints, dependency pinning), which is exactly what you ran into (aiohttp, tavily, langchain_mcp_adapters, missing open_deep_research, Python 3.13.5 constraint). When something breaks, the failure surface becomes larger: errors can come from packaging, transitive dependencies, or mismatched kernels rather than your notebook code. You also lose some immediate visibility: when logic lives in imported files, it’s harder for a learner to see “what happens next” without jumping between files, and stepping through execution in a notebook is less direct than inspecting the cells. Finally, imports can create coupling to the library’s exact structure; if the repo changes layout (e.g., src/ move, renames), notebooks break even if the conceptual workflow is unchanged.

So: importing is the right move for anything you want to reuse, test, or productionize, but it raises the bar on environment hygiene and makes the notebook less self-contained for learning/debugging.

🏗️ Activity #1: Explore the Prompts
Open open_deep_library/prompts.py and examine one of the prompt templates in detail.

Requirements:

Choose one prompt template (clarify, brief, supervisor, researcher, compression, or final report)
Explain what the prompt is designed to accomplish
Identify 2-3 key techniques used in the prompt (e.g., structured output, role definition, examples)
Suggest one improvement you might make to the prompt
YOUR CODE HERE - Write your analysis in a markdown cell below

Prompt deep dive (compress_research_system_prompt)
The compress_research_system_prompt is designed to solve the most common production failure in deep-research agents: the researcher produces a messy, token-heavy trail of tool outputs and partial conclusions, and the downstream report writer either overflows the context window or loses critical evidence. This prompt’s job is not to “summarize” in the normal sense; it is to normalize, deduplicate, and reformat the accumulated research artifacts into a loss-minimized, citation-preserving intermediate report that can be safely merged with other researchers’ outputs and then fed to final report generation. Conceptually, it behaves like a “forensic evidence pack”: it keeps claims intact, keeps sources intact, and makes the research legible without changing its meaning.

Two core techniques are doing most of the work here. First, the prompt uses explicit role + task constraints to prevent the model’s default compression behavior: it repeatedly insists on preserving information “verbatim” and “not losing any sources,” and it frames the task as “clean up” rather than “summarize.” This is important because LLMs will otherwise optimize for brevity and coherence, which is exactly what you don’t want at this stage. Second, it enforces a structured output schema (“List of Queries and Tool Calls Made / Fully Comprehensive Findings / List of All Relevant Sources”), plus hard citation rules (each unique URL gets a single numeric citation, sequential numbering, and a final Sources list). That structure is not aesthetic—it’s an interoperability contract: it makes the output machine-mergeable and auditable, and it prevents the final report model from inventing citations or dropping URLs. A third technique is redundancy-aware normalization: it explicitly allows grouping (“Three sources stated X”), which reduces repetition while still preserving the core statement and associating it with multiple sources—this is a practical compromise between “verbatim everything” and token budgets.

One improvement I would make is to resolve the internal contradiction between “rewrite findings verbatim” and “don’t rewrite it / don’t paraphrase it.” Right now the prompt says “repeated and rewritten verbatim,” then later says “don’t rewrite it, don’t summarize it, don’t paraphrase it.” Those directives conflict and can cause unstable behavior: some runs will over-copy raw tool outputs, others will lightly paraphrase and violate the “verbatim” constraint. A clean fix is to define an explicit transformation policy, for example: “You may only: (a) remove duplicates, (b) reorder into sections, (c) replace pronouns with explicit nouns, (d) shorten only by deleting clearly irrelevant lines, (e) quote key statements exactly as they appear in tool outputs.” That gives the model a bounded edit set, improves determinism, and reduces the chance that critical wording (or citations) shifts during “cleanup.”

🤝 Breakout Room #2
Building & Running the Researcher
In this breakout room, we'll explore the node functions, build the graph, and run wellness research.

Task 6: Node Functions - The Building Blocks
Now let's look at the node functions that make up our graph. We'll import them from the library and understand what each does.

The Complete Research Workflow
The workflow consists of 8 key nodes organized into 3 subgraphs:

Main Graph Nodes:

clarify_with_user - Entry point that checks if clarification is needed
write_research_brief - Transforms user input into structured research brief
final_report_generation - Synthesizes all research into final report
Supervisor Subgraph Nodes:

supervisor - Lead researcher that plans and delegates
supervisor_tools - Executes supervisor's tool calls (delegation, reflection)
Researcher Subgraph Nodes:

researcher - Individual researcher conducting focused research
researcher_tools - Executes researcher's tool calls (search, reflection)
compress_research - Synthesizes researcher's findings
All nodes are defined in open_deep_library/deep_researcher.py

Node 1: clarify_with_user
Purpose: Analyzes user messages and asks clarifying questions if the research scope is unclear.

Key Steps:

Check if clarification is enabled in configuration
Use structured output to analyze if clarification is needed
If needed, end with a clarifying question for the user
If not needed, proceed to research brief with verification message
Implementation: open_deep_library/deep_researcher.py lines 60-115

# Import the clarify_with_user node
from open_deep_library.deep_researcher import clarify_with_user
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[7], line 2
      1 # Import the clarify_with_user node
----> 2 from open_deep_library.deep_researcher import clarify_with_user

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 2: write_research_brief
Purpose: Transforms user messages into a structured research brief for the supervisor.

Key Steps:

Use structured output to generate detailed research brief from messages
Initialize supervisor with system prompt and research brief
Set up supervisor messages with proper context
Why this matters: A well-structured research brief helps the supervisor make better delegation decisions.

Implementation: open_deep_library/deep_researcher.py lines 118-175

# Import the write_research_brief node
from open_deep_library.deep_researcher import write_research_brief
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[8], line 2
      1 # Import the write_research_brief node
----> 2 from open_deep_library.deep_researcher import write_research_brief

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 3: supervisor
Purpose: Lead research supervisor that plans research strategy and delegates to sub-researchers.

Key Steps:

Configure model with three tools:
ConductResearch - Delegate research to a sub-agent
ResearchComplete - Signal that research is done
think_tool - Strategic reflection before decisions
Generate response based on current context
Increment research iteration count
Proceed to tool execution
Decision Making: The supervisor uses think_tool to reflect before delegating research, ensuring thoughtful decomposition of the research question.

Implementation: open_deep_library/deep_researcher.py lines 178-223

# Import the supervisor node (from supervisor subgraph)
from open_deep_library.deep_researcher import supervisor
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[9], line 2
      1 # Import the supervisor node (from supervisor subgraph)
----> 2 from open_deep_library.deep_researcher import supervisor

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 4: supervisor_tools
Purpose: Executes the supervisor's tool calls, including strategic thinking and research delegation.

Key Steps:

Check exit conditions:
Exceeded maximum iterations
No tool calls made
ResearchComplete called
Process think_tool calls for strategic reflection
Execute ConductResearch calls in parallel:
Spawn researcher subgraphs for each delegation
Limit to max_concurrent_research_units (default: 5)
Gather all results asynchronously
Aggregate findings and return to supervisor
Parallel Execution: This is where the magic happens - multiple researchers work simultaneously on different aspects of the research question.

Implementation: open_deep_library/deep_researcher.py lines 225-349

# Import the supervisor_tools node
from open_deep_library.deep_researcher import supervisor_tools
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[10], line 2
      1 # Import the supervisor_tools node
----> 2 from open_deep_library.deep_researcher import supervisor_tools

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 5: researcher
Purpose: Individual researcher that conducts focused research on a specific topic.

Key Steps:

Load all available tools (search, MCP, reflection)
Configure model with tools and researcher system prompt
Generate response with tool calls
Increment tool call iteration count
ReAct Pattern: Researchers use think_tool to reflect after each search, deciding whether to continue or provide their answer.

Available Tools:

Search tools (Tavily or Anthropic native search)
think_tool for strategic reflection
ResearchComplete to signal completion
MCP tools (if configured)
Implementation: open_deep_library/deep_researcher.py lines 365-424

# Import the researcher node (from researcher subgraph)
from open_deep_library.deep_researcher import researcher
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[11], line 2
      1 # Import the researcher node (from researcher subgraph)
----> 2 from open_deep_library.deep_researcher import researcher

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 6: researcher_tools
Purpose: Executes the researcher's tool calls, including searches and strategic reflection.

Key Steps:

Check early exit conditions (no tool calls, native search used)
Execute all tool calls in parallel:
Search tools fetch and summarize web content
think_tool records strategic reflections
MCP tools execute external integrations
Check late exit conditions:
Exceeded max_react_tool_calls (default: 10)
ResearchComplete called
Continue research loop or proceed to compression
Error Handling: Safely handles tool execution errors and continues with available results.

Implementation: open_deep_library/deep_researcher.py lines 435-509

# Import the researcher_tools node
from open_deep_library.deep_researcher import researcher_tools
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[12], line 2
      1 # Import the researcher_tools node
----> 2 from open_deep_library.deep_researcher import researcher_tools

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 7: compress_research
Purpose: Compresses and synthesizes research findings into a concise, structured summary.

Key Steps:

Configure compression model
Add compression instruction to messages
Attempt compression with retry logic:
If token limit exceeded, remove older messages
Retry up to 3 times
Extract raw notes from tool and AI messages
Return compressed research and raw notes
Why Compression? Researchers may accumulate lots of tool outputs and reflections. Compression ensures:

All important information is preserved
Redundant information is deduplicated
Content stays within token limits for the final report
Token Limit Handling: Gracefully handles token limit errors by progressively truncating messages.

Implementation: open_deep_library/deep_researcher.py lines 511-585

# Import the compress_research node
from open_deep_library.deep_researcher import compress_research
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[13], line 2
      1 # Import the compress_research node
----> 2 from open_deep_library.deep_researcher import compress_research

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Node 8: final_report_generation
Purpose: Generates the final comprehensive research report from all collected findings.

Key Steps:

Extract all notes from completed research
Configure final report model
Attempt report generation with retry logic:
If token limit exceeded, truncate findings by 10%
Retry up to 3 times
Return final report or error message
Token Limit Strategy:

First retry: Use model's token limit × 4 as character limit
Subsequent retries: Reduce by 10% each time
Graceful degradation with helpful error messages
Report Quality: The prompt guides the model to create well-structured reports with:

Proper headings and sections
Inline citations
Comprehensive coverage of all findings
Sources section at the end
Implementation: open_deep_library/deep_researcher.py lines 607-697

# Import the final_report_generation node
from open_deep_library.deep_researcher import final_report_generation
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[14], line 2
      1 # Import the final_report_generation node
----> 2 from open_deep_library.deep_researcher import final_report_generation

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Task 7: Graph Construction - Putting It All Together
The system is organized into three interconnected graphs:

1. Researcher Subgraph (Bottom Level)
Handles individual focused research on a specific topic:

START → researcher → researcher_tools → compress_research → END
               ↑            ↓
               └────────────┘ (loops until max iterations or ResearchComplete)
2. Supervisor Subgraph (Middle Level)
Manages research delegation and coordination:

START → supervisor → supervisor_tools → END
            ↑              ↓
            └──────────────┘ (loops until max iterations or ResearchComplete)
            
supervisor_tools spawns multiple researcher_subgraphs in parallel
3. Main Deep Researcher Graph (Top Level)
Orchestrates the complete research workflow:

START → clarify_with_user → write_research_brief → research_supervisor → final_report_generation → END
                 ↓                                       (supervisor_subgraph)
               (may end early if clarification needed)
Let's import the compiled graphs from the library.

# Import the pre-compiled graphs from the library
from open_deep_library.deep_researcher import (
    # Bottom level: Individual researcher workflow
    researcher_subgraph,    # Lines 588-605: researcher → researcher_tools → compress_research
    
    # Middle level: Supervisor coordination
    supervisor_subgraph,    # Lines 351-363: supervisor → supervisor_tools (spawns researchers)
    
    # Top level: Complete research workflow
    deep_researcher,        # Lines 699-719: Main graph with all phases
)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[15], line 2
      1 # Import the pre-compiled graphs from the library
----> 2 from open_deep_library.deep_researcher import (
      3     # Bottom level: Individual researcher workflow
      4     researcher_subgraph,    # Lines 588-605: researcher → researcher_tools → compress_research
      5 
      6     # Middle level: Supervisor coordination
      7     supervisor_subgraph,    # Lines 351-363: supervisor → supervisor_tools (spawns researchers)
      8 
      9     # Top level: Complete research workflow
     10     deep_researcher,        # Lines 699-719: Main graph with all phases
     11 )

File ~/Documents/AI Makerspace/AIE9/08_Open_DeepResearch/open_deep_library/deep_researcher.py:19
     16 from langgraph.graph import END, START, StateGraph
     17 from langgraph.types import Command
---> 19 from open_deep_research.configuration import (
     20     Configuration,
     21 )
     22 from open_deep_research.prompts import (
     23     clarify_with_user_instructions,
     24     compress_research_simple_human_message,
   (...)     29     transform_messages_into_research_topic_prompt,
     30 )
     31 from open_deep_research.state import (
     32     AgentInputState,
     33     AgentState,
   (...)     40     SupervisorState,
     41 )

ModuleNotFoundError: No module named 'open_deep_research'
Why This Architecture?
Advantages of Supervisor-Researcher Delegation
Dynamic Task Decomposition

Unlike section-based approaches with predefined structure, the supervisor can break down research based on the actual question
Adapts to different types of research (comparisons, lists, deep dives, etc.)
Parallel Execution

Multiple researchers work simultaneously on different aspects
Much faster than sequential section processing
Configurable parallelism (1-20 concurrent researchers)
ReAct Pattern for Quality

Researchers use think_tool to reflect after each search
Prevents excessive searching and improves search quality
Natural stopping conditions based on information sufficiency
Flexible Tool Integration

Easy to add MCP tools for specialized research
Supports multiple search APIs (Anthropic, Tavily)
Each researcher can use different tool combinations
Graceful Token Limit Handling

Compression prevents token overflow
Progressive truncation in final report generation
Research can scale to arbitrary depths
Trade-offs
Complexity: More moving parts than section-based approach
Cost: Parallel researchers use more tokens (but faster)
Unpredictability: Research structure emerges dynamically
Task 8: Running the Deep Researcher
Now let's see the system in action! We'll use it to research wellness strategies for improving sleep quality.

Setup
We need to:

Set up the wellness research request
Configure the execution with Anthropic settings
Run the research workflow
# Set up the graph with Anthropic configuration
from IPython.display import Markdown, display
import uuid

# Note: deep_researcher is already compiled from the library
# For this demo, we'll use it directly without additional checkpointing
graph = deep_researcher

print("✓ Graph ready for execution")
print("  (Note: The graph is pre-compiled from the library)")
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[16], line 7
      3 import uuid
      5 # Note: deep_researcher is already compiled from the library
      6 # For this demo, we'll use it directly without additional checkpointing
----> 7 graph = deep_researcher
      9 print("✓ Graph ready for execution")
     10 print("  (Note: The graph is pre-compiled from the library)")

NameError: name 'deep_researcher' is not defined
Configuration for Anthropic
We'll configure the system to use:

Claude Sonnet 4 for all research, supervision, and report generation
Tavily for web search (you can also use Anthropic's native search)
Moderate parallelism (1 concurrent researcher for cost control)
Clarification enabled (will ask if research scope is unclear)
# Configure for Anthropic with moderate settings
config = {
    "configurable": {
        # Model configuration - using Claude Sonnet 4 for everything
        "research_model": "anthropic:claude-sonnet-4-20250514",
        "research_model_max_tokens": 10000,
        
        "compression_model": "anthropic:claude-sonnet-4-20250514",
        "compression_model_max_tokens": 8192,
        
        "final_report_model": "anthropic:claude-sonnet-4-20250514",
        "final_report_model_max_tokens": 10000,
        
        "summarization_model": "anthropic:claude-sonnet-4-20250514",
        "summarization_model_max_tokens": 8192,
        
        # Research behavior
        "allow_clarification": True,
        "max_concurrent_research_units": 1,  # 1 parallel researcher
        "max_researcher_iterations": 2,      # Supervisor can delegate up to 2 times
        "max_react_tool_calls": 3,           # Each researcher can make up to 3 tool calls
        
        # Search configuration
        "search_api": "tavily",  # Using Tavily for web search
        "max_content_length": 50000,
        
        # Thread ID for this conversation
        "thread_id": str(uuid.uuid4())
    }
}

print("✓ Configuration ready")
print(f"  - Research Model: Claude Sonnet 4")
print(f"  - Max Concurrent Researchers: 1")
print(f"  - Max Iterations: 2")
print(f"  - Search API: Tavily")
✓ Configuration ready
  - Research Model: Claude Sonnet 4
  - Max Concurrent Researchers: 1
  - Max Iterations: 2
  - Search API: Tavily
Execute the Wellness Research
Now let's run the research! We'll ask the system to research evidence-based strategies for improving sleep quality.

The workflow will:

Clarify - Check if the request is clear (may skip if obvious)
Research Brief - Transform our request into a structured brief
Supervisor - Plan research strategy and delegate to researchers
Parallel Research - Researchers gather information simultaneously
Compression - Each researcher synthesizes their findings
Final Report - All findings combined into comprehensive report
# Create our wellness research request
research_request = """
I want to improve my sleep quality. I currently:
- Go to bed at inconsistent times (10pm-1am)
- Use my phone in bed
- Often feel tired in the morning

Please research the best evidence-based strategies for improving sleep quality and create a comprehensive sleep improvement plan for me.
"""

# Execute the graph
async def run_research():
    """Run the research workflow and display results."""
    print("Starting research workflow...\n")
    
    async for event in graph.astream(
        {"messages": [{"role": "user", "content": research_request}]},
        config,
        stream_mode="updates"
    ):
        # Display each step
        for node_name, node_output in event.items():
            print(f"\n{'='*60}")
            print(f"Node: {node_name}")
            print(f"{'='*60}")
            
            if node_name == "clarify_with_user":
                if "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    print(f"\n{last_msg.content}")
            
            elif node_name == "write_research_brief":
                if "research_brief" in node_output:
                    print(f"\nResearch Brief Generated:")
                    print(f"{node_output['research_brief'][:500]}...")
            
            elif node_name == "supervisor":
                print(f"\nSupervisor planning research strategy...")
                if "supervisor_messages" in node_output:
                    last_msg = node_output["supervisor_messages"][-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"Tool calls: {len(last_msg.tool_calls)}")
                        for tc in last_msg.tool_calls:
                            print(f"  - {tc['name']}")
            
            elif node_name == "supervisor_tools":
                print(f"\nExecuting supervisor's tool calls...")
                if "notes" in node_output:
                    print(f"Research notes collected: {len(node_output['notes'])}")
            
            elif node_name == "final_report_generation":
                if "final_report" in node_output:
                    print(f"\n" + "="*60)
                    print("FINAL REPORT GENERATED")
                    print("="*60 + "\n")
                    display(Markdown(node_output["final_report"]))
    
    print("\n" + "="*60)
    print("Research workflow completed!")
    print("="*60)

# Run the research
await run_research()
Starting research workflow...

---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[18], line 63
     60     print("="*60)
     62 # Run the research
---> 63 await run_research()

Cell In[18], line 16, in run_research()
     13 """Run the research workflow and display results."""
     14 print("Starting research workflow...\n")
---> 16 async for event in graph.astream(
     17     {"messages": [{"role": "user", "content": research_request}]},
     18     config,
     19     stream_mode="updates"
     20 ):
     21     # Display each step
     22     for node_name, node_output in event.items():
     23         print(f"\n{'='*60}")

NameError: name 'graph' is not defined
Task 9: Understanding the Output
Let's break down what happened:

Phase 1: Clarification
The system checked if your request was clear. Since you provided specific details about your sleep issues, it likely proceeded without asking clarifying questions.

Phase 2: Research Brief
Your request was transformed into a detailed research brief that guides the supervisor's delegation strategy.

Phase 3: Supervisor Delegation
The supervisor analyzed the brief and decided how to break down the research:

Used think_tool to plan strategy
Called ConductResearch to delegate to researchers
Each delegation specified a focused research topic (e.g., sleep hygiene, circadian rhythm, blue light effects)
Phase 4: Parallel Research
Researchers worked on their assigned topics:

Each researcher used web search tools to gather information
Used think_tool to reflect after each search
Decided when they had enough information
Compressed their findings into clean summaries
Phase 5: Final Report
All research findings were synthesized into a comprehensive sleep improvement plan with:

Well-structured sections
Evidence-based recommendations
Practical action items
Sources for further reading
Task 10: Key Takeaways & Next Steps
Architecture Benefits
Dynamic Decomposition - Research structure emerges from the question, not predefined
Parallel Efficiency - Multiple researchers work simultaneously
ReAct Quality - Strategic reflection improves search decisions
Scalability - Handles token limits gracefully through compression
Flexibility - Easy to add new tools and capabilities
When to Use This Pattern
Complex research questions that need multi-angle investigation
Comparison tasks where parallel research on different topics is beneficial
Open-ended exploration where structure should emerge dynamically
Time-sensitive research where parallel execution speeds up results
When to Use Section-Based Instead
Highly structured reports with predefined format requirements
Template-based content where sections are always the same
Sequential dependencies where later sections depend on earlier ones
Budget constraints where token efficiency is critical
Extend the System
Add MCP Tools - Integrate specialized tools for your domain
Custom Prompts - Modify prompts for specific research types
Different Models - Try different Claude versions or mix models
Persistence - Use a real database for checkpointing instead of memory
Learn More
LangGraph Documentation
Open Deep Research Repo
Anthropic Claude Documentation
Tavily Search API
❓ Question #3:
What are the trade-offs of using parallel researchers vs. sequential research? When might you choose one approach over the other?

Answer:
Using parallel researchers trades higher cost and coordination complexity for speed and coverage. When multiple sub-agents work at the same time, the system can explore independent angles of a question simultaneously—definitions, evidence, counterpoints, and applications—rather than waiting for one line of inquiry to finish before starting the next. This is especially valuable for open-ended or comparative research where dimensions are loosely coupled and can be investigated independently. Parallelism also reduces the risk of early fixation: if one researcher misses an important angle, another may still surface it. The downside is that parallel execution increases token usage and API pressure, can introduce redundancy across researchers, and requires careful aggregation and compression to avoid overwhelming the final report stage. It also makes the system harder to reason about and debug because multiple partial truths arrive asynchronously.

Sequential research optimizes for efficiency, coherence, and cost control. Each step builds on the previous one, allowing later searches to be guided by what is already known. This is well-suited for tightly coupled questions where later inquiry depends on earlier findings (e.g., diagnosing a problem before researching solutions, or drilling progressively deeper into a single technical mechanism). Sequential flows are easier to debug, cheaper in token usage, and less likely to produce overlapping or conflicting findings. However, they are slower end-to-end and more vulnerable to early bias: if the initial search direction is weak or incomplete, the entire research trajectory may suffer.

In practice, you choose parallel researchers when the problem has multiple independent dimensions, time matters, and breadth of coverage is more important than minimal cost (comparisons, landscape surveys, policy or wellness research). You choose sequential research when the problem requires stepwise reasoning, tight dependency between steps, or strict budget and determinism (technical debugging, protocol design, or deeply focused analysis).

❓ Question #4:
How would you adapt this deep research architecture for a production wellness application? What additional components would you need?

Answer:
To adapt this architecture for a production wellness application, I’d keep the supervisor–researcher delegation pattern but constrain it into a clinically safer, user-specific decision-support workflow. The top-level Agent state would become a durable “wellness case file” that includes a user profile (age range, goals, constraints, preferences), ongoing metrics (sleep duration/efficiency, activity, stress), and a history of prior plans and outcomes. The supervisor would still decompose questions, but it would do so under a stricter policy: it should first classify intent (education vs. planning vs. symptom triage), detect whether the request is health-sensitive, and decide whether to proceed, ask clarification, or route to a medical escalation path. Researchers would stay as isolated sub-agents, but their tool access would be tightly scoped to approved medical sources, and their outputs would be forced into evidence tiers (guidelines/systematic reviews vs. observational studies vs. blogs) before anything reaches the plan generator.

You’d need additional components beyond the notebook to make it production-grade. First is a data and identity layer: secure user authentication, consent, data minimization, and encrypted storage for profile and logs (HIPAA considerations if you’re handling PHI in the U.S.). Second is a grounding and safety layer: curated source allowlists (CDC/NIH, professional society guidelines, Cochrane, peer-reviewed journals), citation verification, and a medical safety classifier that detects red flags (e.g., sleep apnea symptoms, severe depression, chest pain) and switches the agent from “plan” to “seek professional help.” Third is an evaluation and quality system: automated tests for hallucination/citation mismatch, plan adherence to constraints, and regression tracking; plus human review workflows for prompt changes and high-risk outputs. Fourth is observability and controls: structured traces of supervisor decisions, tool calls, token/cost budgets, caching, rate-limit handling, and “why this recommendation” explanations so the product is debuggable and trustworthy.

Finally, the output side needs to be product-shaped, not just a report. You’d add a plan compiler that turns research findings into an actionable, personalized program with scheduling, reminders, habit loops, and measurable checkpoints, plus a feedback loop that updates the plan based on outcomes (did sleep latency improve? did the user comply?). This is where long-term memory becomes useful: store what interventions were tried, what worked, what failed, and adapt the next iteration. In short: keep the delegation and compression mechanics, but add privacy/security, clinical safety gating, trusted-source retrieval, evaluation/monitoring, and a personalization+feedback subsystem so the agent behaves like a safe, measurable wellness coach rather than a generic research bot.

🏗️ Activity #2: Custom Wellness Research
Using what you've learned, run a custom wellness research task.

Requirements:

Create a wellness-related research question (exercise, nutrition, stress, etc.)
Modify the configuration for your use case
Run the research and analyze the output
Document what worked well and what could be improved
Experiment ideas:

Research exercise routines for specific conditions (bad knee, lower back pain)
Compare different stress management techniques
Investigate nutrition strategies for specific goals
Explore meditation and mindfulness research
YOUR CODE HERE

Activity #2 — Custom Wellness Research: Strength training plan for lower back pain (evidence-based)

Wellness research question: We want an evidence-based, beginner-friendly strength + mobility plan to reduce recurring lower back pain from prolonged sitting. We want safe exercises, progression, and red flags (when to stop / seek care). I prefer minimal equipment (bodyweight + resistance band), 20–30 minutes per session, 3–4 days/week.

Configuration changes (why): We reduced token limits and capped tool calls to avoid rate limits and excessive context growth. We also set max_content_length lower so page summarization stays bounded. We kept clarification enabled in case key constraints are missing.

What we expect from the system: The supervisor should delegate to 1 researcher (simple scope), gather guidelines + reputable sources, then produce a structured plan with progression and safety constraints.

import uuid
from IPython.display import Markdown, display

# 1) Custom wellness request
my_wellness_request = """
I have recurring mild-to-moderate lower back pain that seems related to prolonged sitting.
I want an evidence-based exercise plan to improve back resilience and reduce pain.

Constraints:
- Beginner-friendly
- Minimal equipment (bodyweight + resistance band only)
- 20–30 minutes per session
- 3–4 days per week
- Include mobility + strengthening
- Include progression guidance (how to increase difficulty over 4–6 weeks)
- Include safety notes and red flags (when to stop and seek medical care)

Please research evidence-based recommendations (prefer reputable medical sources, guidelines, or systematic reviews)
and create a clear 4–6 week plan with specific exercises, sets/reps, and weekly progression.
"""

# 2) Safer config to reduce rate-limit + token pressure
my_config = {
    "configurable": {
        # Models: keep same, but lower max tokens to reduce TPM risk
        "research_model": "anthropic:claude-sonnet-4-20250514",
        "research_model_max_tokens": 5000,

        "compression_model": "anthropic:claude-sonnet-4-20250514",
        "compression_model_max_tokens": 4096,

        "final_report_model": "anthropic:claude-sonnet-4-20250514",
        "final_report_model_max_tokens": 6000,

        "summarization_model": "anthropic:claude-sonnet-4-20250514",
        "summarization_model_max_tokens": 2048,

        # Behavior: keep it simple to avoid runaway tool usage
        "allow_clarification": True,
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": 1,   # supervisor delegates once
        "max_react_tool_calls": 2,        # researcher does 1–2 searches max

        # Search: keep content smaller to reduce summarization timeouts
        "search_api": "tavily",
        "max_content_length": 20000,

        "thread_id": str(uuid.uuid4())
    }
}

# 3) Runner (reuse your earlier pattern, but simplified)
async def run_custom_research(request: str, cfg: dict):
    async for event in graph.astream(
        {"messages": [{"role": "user", "content": request}]},
        cfg,
        stream_mode="updates"
    ):
        for node_name, node_output in event.items():
            print(f"\n{'='*70}\nNode: {node_name}\n{'='*70}")

            if node_name == "clarify_with_user" and "messages" in node_output:
                print(node_output["messages"][-1].content)

            if node_name == "write_research_brief" and "research_brief" in node_output:
                print("\nResearch brief (first 400 chars):\n")
                print(node_output["research_brief"][:400] + "...")

            if node_name in ("supervisor", "research_supervisor") and "supervisor_messages" in node_output:
                last_msg = node_output["supervisor_messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print("\nSupervisor tool calls:")
                    for tc in last_msg.tool_calls:
                        print(f" - {tc['name']}")

            if node_name == "supervisor_tools" and "notes" in node_output:
                print(f"\nNotes collected: {len(node_output['notes'])}")

            if node_name == "final_report_generation" and "final_report" in node_output:
                print("\n" + "="*70)
                print("FINAL REPORT")
                print("="*70 + "\n")
                display(Markdown(node_output["final_report"]))

# 4) Execute
await run_custom_research(my_wellness_request, my_config)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[19], line 86
     83                 display(Markdown(node_output["final_report"]))
     85 # 4) Execute
---> 86 await run_custom_research(my_wellness_request, my_config)

Cell In[19], line 54, in run_custom_research(request, cfg)
     53 async def run_custom_research(request: str, cfg: dict):
---> 54     async for event in graph.astream(
     55         {"messages": [{"role": "user", "content": request}]},
     56         cfg,
     57         stream_mode="updates"
     58     ):
     59         for node_name, node_output in event.items():
     60             print(f"\n{'='*70}\nNode: {node_name}\n{'='*70}")

NameError: name 'graph' is not defined