import streamlit as st
from rag_agent import ask

st.set_page_config(page_title="LearnForge", layout="wide")

# ----------------------------
# Session state setup
# ----------------------------
if "projects" not in st.session_state:
    st.session_state.projects = [
        {
            "name": "Research AI",
            "description": "Notes, frameworks, and applied research workflows for AI systems.",
            "files": ["agent_design.md", "evaluation_notes.md"],
        },
        {
            "name": "ML Papers",
            "description": "Paper summaries, architecture notes, and benchmark learnings.",
            "files": ["attention_is_all_you_need.md", "ragas_notes.md"],
        },
        {
            "name": "LangGraph Deep Agents",
            "description": "Concepts around planning, memory, subagents, and long-horizon tasks.",
            "files": ["deep_agents_session.md"],
        },
        {
            "name": "RAG Evaluation",
            "description": "Golden datasets, retrieval metrics, grounding, and answer quality experiments.",
            "files": ["rag_eval_checklist.md", "results_comparison.csv"],
        },
        {
            "name": "Prompt Engineering",
            "description": "Prompt templates, structured outputs, and iterative prompt testing notes.",
            "files": ["prompt_patterns.md"],
        },
    ]

if "selected_project" not in st.session_state:
    st.session_state.selected_project = st.session_state.projects[0]["name"]

if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

if "embeddings_generated_for" not in st.session_state:
    st.session_state.embeddings_generated_for = ""

if "skill_test_started" not in st.session_state:
    st.session_state.skill_test_started = False

if "skill_test_submitted" not in st.session_state:
    st.session_state.skill_test_submitted = False

if "selected_quiz_type" not in st.session_state:
    st.session_state.selected_quiz_type = "Quick Recall"

if "selected_difficulty" not in st.session_state:
    st.session_state.selected_difficulty = "Advanced"

# ----------------------------
# Helper functions
# ----------------------------
def get_questions_and_keywords(project_name: str, quiz_type: str, difficulty: str):
    """
    Returns:
      questions: list[str]
      keyword_sets: list[list[str]]
    """

    is_deep_agents = "deep agents" in project_name.lower()

    if is_deep_agents:
        if quiz_type == "Quick Recall":
            if difficulty == "Beginner":
                questions = [
                    "What are the four core components of a Deep Agent system?",
                    "What is the purpose of planning in a Deep Agent?",
                    "What is long-term memory used for in agent systems?",
                ]
                keyword_sets = [
                    ["planning", "context", "subagent", "memory"],
                    ["task", "todo", "plan", "progress"],
                    ["store", "reuse", "memory", "future"],
                ]
            elif difficulty == "Intermediate":
                questions = [
                    "Why does a Deep Agent maintain a persistent task list?",
                    "What does context management do in a Deep Agent system?",
                    "Why would a Deep Agent spawn subagents?",
                ]
                keyword_sets = [
                    ["persistent", "task", "todo", "multi-step", "progress"],
                    ["context", "retrieve", "relevant", "organize", "information"],
                    ["delegate", "specialized", "subagent", "parallel", "focused"],
                ]
            else:
                questions = [
                    "Name the four core components of a Deep Agent architecture.",
                    "What problem does context compression help solve?",
                    "What architectural benefit comes from separating memory namespaces by project?",
                ]
                keyword_sets = [
                    ["planning", "context", "subagent", "memory"],
                    ["compression", "token", "context", "overflow", "summarize"],
                    ["project", "namespace", "separate", "memory", "retrieve"],
                ]

        elif quiz_type == "Concept Check":
            if difficulty == "Beginner":
                questions = [
                    "Explain planning in a Deep Agent in simple terms.",
                    "Explain subagent spawning in simple terms.",
                    "Why is memory useful in an intelligent agent?",
                ]
                keyword_sets = [
                    ["planning", "task", "steps", "track"],
                    ["subagent", "specialized", "delegate"],
                    ["memory", "store", "reuse", "learn"],
                ]
            elif difficulty == "Intermediate":
                questions = [
                    "How do planning and context management work together in a Deep Agent?",
                    "Why is subagent spawning important for specialized workflows?",
                    "How does long-term memory improve future reasoning?",
                ]
                keyword_sets = [
                    ["planning", "context", "relevant", "task", "organize"],
                    ["subagent", "specialized", "delegate", "complex"],
                    ["long-term", "memory", "reuse", "future", "reasoning"],
                ]
            else:
                questions = [
                    "Explain how planning, context management, subagent spawning, and long-term memory complement each other in a Deep Agent.",
                    "Why is context compression important in long-horizon agent systems?",
                    "Why is long-term memory different from short conversational context?",
                ]
                keyword_sets = [
                    ["planning", "context", "subagent", "memory", "together"],
                    ["compression", "token", "long-horizon", "summarize", "context"],
                    ["long-term", "short", "memory", "persistent", "session"],
                ]

        else:  # Applied Reasoning
            if difficulty == "Beginner":
                questions = [
                    "A user uploads new notes into a Deep Agents project. What should the system do next?",
                    "If an agent gets too much context, what is one thing the system can do?",
                    "If a task has research and writing parts, how can subagents help?",
                ]
                keyword_sets = [
                    ["embed", "index", "retrieve", "project"],
                    ["compress", "summarize", "context"],
                    ["subagent", "research", "writing", "delegate"],
                ]
            elif difficulty == "Intermediate":
                questions = [
                    "How would you design a workflow where a Deep Agent plans tasks, retrieves relevant knowledge, and delegates research to a subagent?",
                    "How would you avoid irrelevant retrieval in a system with multiple knowledge projects?",
                    "How would memory help if the user returns later and continues the same learning project?",
                ]
                keyword_sets = [
                    ["plan", "retrieve", "subagent", "research", "workflow"],
                    ["project", "filter", "namespace", "relevant", "retrieve"],
                    ["memory", "return", "continue", "store", "profile"],
                ]
            else:
                questions = [
                    "Design a Deep Agent workflow for long-horizon learning support across multiple sessions.",
                    "How would you extend Deep Agents to support multiple project-specific indexes without contaminating retrieval across projects?",
                    "How would you evaluate whether the Deep Agent is improving learning outcomes over time?",
                ]
                keyword_sets = [
                    ["planning", "memory", "subagent", "context", "sessions"],
                    ["project", "index", "namespace", "separate", "retrieval"],
                    ["evaluate", "improve", "outcomes", "metrics", "progress"],
                ]
    else:
        if quiz_type == "Quick Recall":
            questions = [
                "What is the main topic of this project?",
                "Name one important concept from this project.",
                "What problem does this project help solve?",
            ]
            keyword_sets = [
                ["topic", "project"],
                ["concept", "key"],
                ["problem", "solve"],
            ]
        elif quiz_type == "Concept Check":
            questions = [
                "Explain one key concept from this project.",
                "Why is this concept important?",
                "How would you describe this project to another learner?",
            ]
            keyword_sets = [
                ["concept", "project"],
                ["important", "value"],
                ["describe", "learner"],
            ]
        else:
            questions = [
                "How would you apply this project in a practical setting?",
                "How would you improve the system behind this project?",
                "How would you measure success for this project?",
            ]
            keyword_sets = [
                ["apply", "practical"],
                ["improve", "system"],
                ["measure", "success"],
            ]

    return questions, keyword_sets


def calculate_score(answers, keyword_sets):
    total = len(keyword_sets)
    points = 0

    for ans, keywords in zip(answers, keyword_sets):
        text = ans.strip().lower()
        if not text:
            continue

        matched = sum(1 for kw in keywords if kw in text)

        if matched >= 3:
            points += 1
        elif matched == 2:
            points += 0.75
        elif matched == 1:
            points += 0.5

    return round(points, 2), total


# ----------------------------
# Header
# ----------------------------
st.title("LearnForge — Agentic RAG Knowledge Architect")
st.write(
    "Turn personal learning materials into structured knowledge projects. "
    "Query your indexed content, simulate project creation and file uploads, "
    "and showcase the end-to-end learning workflow."
)
st.caption(
    "LearnForge converts personal learning materials into structured knowledge projects "
    "and enables retrieval-driven learning with evaluation."
)

# ----------------------------
# Layout
# ----------------------------
left_col, right_col = st.columns([1, 2], gap="large")

current_project = None

# ============================
# LEFT COLUMN: Project UI
# ============================
with left_col:
    st.subheader("Knowledge Projects")

    project_names = [p["name"] for p in st.session_state.projects]
    selected_project = st.radio(
        "Select a project",
        project_names,
        index=project_names.index(st.session_state.selected_project)
        if st.session_state.selected_project in project_names
        else 0,
    )
    st.session_state.selected_project = selected_project

    current_project = next(
        (p for p in st.session_state.projects if p["name"] == st.session_state.selected_project),
        None,
    )

    if current_project:
        st.markdown(f"### {current_project['name']}")
        st.write(current_project["description"])

        st.markdown("**Files in project**")
        if current_project["files"]:
            for f in current_project["files"]:
                st.write(f"• {f}")
        else:
            st.write("No files yet.")

    st.divider()

    st.subheader("Create New Project")

    with st.form("create_project_form", clear_on_submit=True):
        new_project_name = st.text_input("Project name")
        new_project_description = st.text_area("Project description")
        create_project = st.form_submit_button("Create Project")

        if create_project:
            if not new_project_name.strip():
                st.warning("Please enter a project name.")
            elif new_project_name.strip() in [p["name"] for p in st.session_state.projects]:
                st.warning("A project with this name already exists.")
            else:
                st.session_state.projects.append(
                    {
                        "name": new_project_name.strip(),
                        "description": new_project_description.strip()
                        or "New knowledge project for uploaded learning materials.",
                        "files": [],
                    }
                )
                st.session_state.selected_project = new_project_name.strip()
                st.success(f"Project '{new_project_name.strip()}' created.")

    st.divider()

    st.subheader("Upload Files")

    uploaded_files = st.file_uploader(
        "Upload notes, markdowns, PDFs, or study files",
        accept_multiple_files=True,
        type=None,
    )

    if st.button("Add Files to Project"):
        if not uploaded_files:
            st.info("Select one or more files first.")
        elif current_project is None:
            st.warning("Please select a project.")
        else:
            added_names = [f.name for f in uploaded_files]
            current_project["files"].extend(added_names)
            st.session_state.last_uploaded_files = added_names
            st.success(
                f"Added {len(added_names)} file(s) to '{current_project['name']}'."
            )

    if st.button("Generate Embeddings"):
        if current_project is None:
            st.warning("Please select a project.")
        else:
            st.session_state.embeddings_generated_for = current_project["name"]
            st.success(
                f"Embeddings generated for project '{current_project['name']}'."
            )

    st.divider()

    st.subheader("Project Activity")

    if current_project:
        st.write(f"**Selected project:** {current_project['name']}")
        st.write(f"**Description:** {current_project['description']}")
        st.write(f"**Files currently attached:** {len(current_project['files'])}")

        if st.session_state.last_uploaded_files:
            st.write("**Most recently uploaded files:**")
            for f in st.session_state.last_uploaded_files:
                st.write(f"• {f}")

        if st.session_state.embeddings_generated_for:
            st.write(
                f"**Last embedding trigger:** {st.session_state.embeddings_generated_for}"
            )
    else:
        st.write("No project selected.")

# ============================
# RIGHT COLUMN: Search + Skills
# ============================
with right_col:
    st.subheader("Ask Your Knowledge Base")

    if current_project:
        st.caption(f"Current project: {current_project['name']}")

    q = st.text_input("Your question")

    if st.button("Ask") and q.strip():
        with st.spinner("Thinking..."):
            result = ask(q.strip())

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Retrieved KB chunks (debug)"):
            if not result["retrieved_docs"]:
                st.write("No KB chunks retrieved.")
            else:
                for i, d in enumerate(result["retrieved_docs"], start=1):
                    st.markdown(
                        f"**Chunk {i} — source: {d.metadata.get('source', 'unknown')}**"
                    )
                    st.write(
                        d.page_content[:1200]
                        + ("..." if len(d.page_content) > 1200 else "")
                    )

        with st.expander("Web snippets (if used)"):
            st.write(result["web_snippets"] or "(No web snippets used.)")

    st.divider()

    st.subheader("Test Your Skills")

    quiz_type = st.selectbox(
        "Choose quiz type",
        ["Quick Recall", "Concept Check", "Applied Reasoning"],
        index=["Quick Recall", "Concept Check", "Applied Reasoning"].index(
            st.session_state.selected_quiz_type
        ),
    )
    st.session_state.selected_quiz_type = quiz_type

    difficulty = st.selectbox(
        "Select difficulty",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(
            st.session_state.selected_difficulty
        ),
    )
    st.session_state.selected_difficulty = difficulty

    if st.button("Start Skill Test"):
        st.session_state.skill_test_started = True
        st.session_state.skill_test_submitted = False

    if st.session_state.skill_test_started:
        st.markdown("### Knowledge Assessment")

        if current_project:
            st.caption(f"Project: {current_project['name']}")

        questions, keyword_sets = get_questions_and_keywords(
            current_project["name"] if current_project else "",
            quiz_type,
            difficulty,
        )

        with st.form("knowledge_skill_test"):
            learner_name = st.text_input("Learner name")
            answers = []

            for i, question in enumerate(questions, start=1):
                answer = st.text_area(f"Question {i}: {question}", height=120)
                answers.append(answer)

            submit = st.form_submit_button("Submit")

            if submit:
                st.session_state.skill_test_submitted = True
                score, total = calculate_score(answers, keyword_sets)

                st.success("Skill test submitted successfully.")
                st.write("### Results")
                st.write(f"Score: {score} / {total}")

                if score >= total * 0.85:
                    st.write("Performance: Excellent")
                elif score >= total * 0.6:
                    st.write("Performance: Good")
                elif score > 0:
                    st.write("Performance: Developing")
                else:
                    st.write("Performance: Not enough information provided")