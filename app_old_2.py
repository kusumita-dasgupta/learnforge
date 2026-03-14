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

# ----------------------------
# Header
# ----------------------------
st.title("LearnForge — Agentic RAG Knowledge Architect")
st.write(
    "Turn personal learning materials into structured knowledge projects. "
    "Query your indexed content, simulate project creation and file uploads, "
    "and showcase the end-to-end learning workflow."
)

# ----------------------------
# Layout
# ----------------------------
left_col, right_col = st.columns([1, 2], gap="large")

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
                f"Added {len(added_names)} file(s) to '{current_project['name']}'. "
                "This is a demo UI flow only."
            )

    if st.button("Generate Embeddings"):
        if current_project is None:
            st.warning("Please select a project.")
        else:
            st.session_state.embeddings_generated_for = current_project["name"]
            st.success(
                f"Embeddings generated for project '{current_project['name']}' "
                "(demo trigger only)."
            )

    st.divider()

    st.subheader("Test Your Skills")

    skill_mode = st.selectbox(
        "Choose quiz style",
        ["Quick Recall", "Concept Check", "Applied Reasoning"],
    )

    if st.button("Start Skill Test"):
        if current_project is None:
            st.warning("Please select a project.")
        else:
            st.info(
                f"Launching '{skill_mode}' test for '{current_project['name']}' "
                "(demo flow only)."
            )

# ============================
# RIGHT COLUMN: Existing Q&A
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