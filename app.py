import streamlit as st
from rag_agent import ask

st.set_page_config(page_title="LearnForge", layout="wide")
st.title("LearnForge — Agentic RAG Knowledge Architect")

st.write("Ask questions over your personal KB (Sessions 1–11) with optional web search via Tavily.")

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
                st.markdown(f"**Chunk {i} — source: {d.metadata.get('source','unknown')}**")
                st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))

    with st.expander("Web snippets (if used)"):
        st.write(result["web_snippets"] or "(No web snippets used.)")