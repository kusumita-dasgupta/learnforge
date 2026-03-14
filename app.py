import streamlit as st
from rag_agent import ask

st.set_page_config(page_title="LearnForge", layout="wide")

# ----------------------------
# Safe visual styling
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
    }

    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 1.4rem;
        max-width: 1400px;
    }

    /* Hide Streamlit chrome */
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    [data-testid="stStatusWidget"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
        border: 1px solid #dbe7f3;
        border-radius: 22px;
        padding: 18px 22px 14px 22px;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.04;
        color: #172554;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .subtext {
        font-size: 0.98rem;
        color: #475569;
        line-height: 1.45;
        margin-bottom: 0.6rem;
        max-width: 1080px;
    }

    .chip-row {
        margin-top: 0.25rem;
        margin-bottom: 0;
    }

    /* Make these look like labels, not buttons */
    .metric-chip {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        border-radius: 8px;
        background: #eaf3ff;
        color: #2454d3;
        font-size: 0.84rem;
        font-weight: 700;
        margin-right: 0.38rem;
        margin-bottom: 0.25rem;
        border: none;
        box-shadow: none;
    }

    .card-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.65rem;
    }

    .project-selected-card {
        background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
        border: 2px solid #93c5fd;
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.08);
    }

    .project-selected-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .project-item-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.12rem;
    }

    .project-item-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }

    .project-list-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.55rem;
    }

    .answer-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #d8e6f6;
        border-left: 5px solid #60a5fa;
        border-radius: 14px;
        padding: 16px 16px;
        color: #0f172a;
        margin-top: 10px;
        line-height: 1.62;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.06);
    }

    .results-box {
        background: #f8fbff;
        border: 1px solid #d8e6f6;
        border-radius: 14px;
        padding: 14px;
        margin-top: 8px;
    }

    .feedback-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px;
    }

    .kpi-line {
        font-size: 0.94rem;
        color: #334155;
        line-height: 1.55;
    }

    div[data-testid="stForm"] {
        background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.03);
    }

    div.stButton > button {
        background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
        color: #0f172a;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.38rem 0.9rem;
        white-space: nowrap !important;
        writing-mode: horizontal-tb !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(180deg, #dbeafe 0%, #bfdbfe 100%);
        color: #0f172a;
        border: 1px solid #93c5fd;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 12px !important;
    }

    /* compact action buttons in project list */
    .project-action-note {
        font-size: 0.78rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

        else:
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


def get_feedback(score, total, project_name, quiz_type, difficulty):
    if score >= total * 0.85:
        return "Excellent coverage of the core concepts. Your answers are clear, relevant, and aligned with the selected project."

    if score >= total * 0.6:
        return (
            "Good response quality overall. To improve further, make the answers more specific by naming the key architecture elements, "
            "explaining how they work together, and using stronger project-specific terminology."
        )

    if score > 0:
        if "deep agents" in project_name.lower():
            return (
                "Your answers show partial understanding, but they need stronger technical specificity. "
                "To improve, explicitly mention concepts such as planning, context management, subagent spawning, long-term memory, "
                "project-specific retrieval, evaluation metrics, and multi-session workflows where relevant."
            )
        return (
            "Your answers are partially correct, but they need more concrete detail. "
            "To improve, refer directly to the main concepts of the project, explain how the components work together, "
            "and use more precise terminology instead of broad general statements."
        )

    return (
        "Not enough information was provided to evaluate the responses. "
        "Try answering each question directly and include key concepts from the selected project."
    )


# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">LearnForge — Your Knowledge Architect</div>
        <div class="subtext">
            Turn personal learning materials into structured knowledge projects.
            Query indexed content, organize learning by project, and assess understanding
            through interactive knowledge checks.
        </div>
        <div class="chip-row">
            <span class="metric-chip">Project-based learning</span>
            <span class="metric-chip">Retrieval-driven answers</span>
            <span class="metric-chip">Knowledge assessment</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Layout
# ----------------------------
left_col, right_col = st.columns([0.72, 2.28], gap="large")
current_project = next(
    (p for p in st.session_state.projects if p["name"] == st.session_state.selected_project),
    None,
)

# ============================
# LEFT COLUMN
# ============================
with left_col:
    st.markdown('<div class="card-title">Knowledge Projects</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="project-list-label">Select a project workspace</div>',
        unsafe_allow_html=True,
    )

    if current_project:
        st.markdown(
            f"""
            <div class="project-selected-card">
                <div class="project-selected-badge">Selected project</div>
                <div class="project-item-title">{current_project['name']}</div>
                <div class="project-item-subtitle">{current_project['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for idx, project in enumerate(st.session_state.projects):
        item_cols = st.columns([4.6, 1.6])

        with item_cols[0]:
            st.markdown(
                f"""
                <div class="project-item-title">{project['name']}</div>
                <div class="project-item-subtitle">{project['description']}</div>
                """,
                unsafe_allow_html=True,
            )

        with item_cols[1]:
            label = "Active" if project["name"] == st.session_state.selected_project else "Open"
            if st.button(label, key=f"project_btn_{idx}", use_container_width=True):
                st.session_state.selected_project = project["name"]
                st.session_state.skill_test_started = False
                st.session_state.skill_test_submitted = False

    current_project = next(
        (p for p in st.session_state.projects if p["name"] == st.session_state.selected_project),
        None,
    )

    st.markdown("---")

    if current_project:
        st.markdown("**Files in project**")
        if current_project["files"]:
            for f in current_project["files"]:
                st.write(f"• {f}")
        else:
            st.write("No files yet.")

    st.markdown("---")

    st.markdown('<div class="card-title">Create New Project</div>', unsafe_allow_html=True)
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

    st.markdown("---")

    st.markdown('<div class="card-title">Upload Files</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload notes, markdowns, PDFs, or study files",
        accept_multiple_files=True,
        type=None,
    )

    if st.button("Add Files to Project", use_container_width=True):
        if not uploaded_files:
            st.info("Select one or more files first.")
        elif current_project is None:
            st.warning("Please select a project.")
        else:
            added_names = [f.name for f in uploaded_files]
            current_project["files"].extend(added_names)
            st.session_state.last_uploaded_files = added_names
            st.success(f"Added {len(added_names)} file(s) to '{current_project['name']}'.")

    if st.button("Generate Embeddings", use_container_width=True):
        if current_project is None:
            st.warning("Please select a project.")
        else:
            st.session_state.embeddings_generated_for = current_project["name"]
            st.success(f"Embeddings generated for project '{current_project['name']}'.")

    st.markdown("---")

    st.markdown('<div class="card-title">Project Activity</div>', unsafe_allow_html=True)

    if current_project:
        st.markdown(
            f"""
            <div class="results-box">
                <div class="kpi-line"><strong>Selected project:</strong> {current_project['name']}</div>
                <div class="kpi-line"><strong>Description:</strong> {current_project['description']}</div>
                <div class="kpi-line"><strong>Files currently attached:</strong> {len(current_project['files'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.last_uploaded_files:
            st.write("**Most recently uploaded files:**")
            for f in st.session_state.last_uploaded_files:
                st.write(f"• {f}")

        if st.session_state.embeddings_generated_for:
            st.write(f"**Last embedding trigger:** {st.session_state.embeddings_generated_for}")
    else:
        st.write("No project selected.")

# ============================
# RIGHT COLUMN
# ============================
with right_col:
    st.markdown('<div class="card-title">Ask Your Knowledge Base</div>', unsafe_allow_html=True)

    if current_project:
        st.caption(f"Current project: {current_project['name']}")

    q = st.text_input("Your question")

    if st.button("Ask") and q.strip():
        with st.spinner("Thinking..."):
            result = ask(q.strip())

        st.markdown('<div class="card-title">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

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

    st.markdown("---")

    st.markdown('<div class="card-title">Test Your Skills</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="card-title">Knowledge Assessment</div>', unsafe_allow_html=True)

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
                feedback = get_feedback(
                    score,
                    total,
                    current_project["name"] if current_project else "",
                    quiz_type,
                    difficulty,
                )

                st.success("Skill test submitted successfully.")
                st.markdown('<div class="results-box">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Results</div>', unsafe_allow_html=True)
                st.write(f"**Score:** {score} / {total}")

                if score >= total * 0.85:
                    st.write("**Performance:** Excellent")
                elif score >= total * 0.6:
                    st.write("**Performance:** Good")
                elif score > 0:
                    st.write("**Performance:** Partial mastery")
                else:
                    st.write("**Performance:** Insufficient response")

                st.markdown(
                    f"""
                    <div class="feedback-box">
                        <strong>Feedback:</strong><br>
                        {feedback}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)