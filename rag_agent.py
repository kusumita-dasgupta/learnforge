import os
from typing import TypedDict, List, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from tavily import TavilyClient

from langgraph.graph import StateGraph, END

from retrievers import get_dense_retriever, try_get_rerank_retriever

load_dotenv()

class GraphState(TypedDict):
    question: str
    use_web: bool
    retrieved_docs: List[Document]
    web_snippets: str
    answer: str

def should_use_web(question: str, retrieved_docs: List[Document]) -> bool:
    """
    Minimal heuristic:
    - If user asks for "latest/current/today/2026" -> web
    - If retrieval is empty -> web
    """
    q = question.lower()
    if any(k in q for k in ["latest", "current", "today", "this week", "2026", "newest"]):
        return True
    if not retrieved_docs:
        return True
    return False

def retrieve_node(state: GraphState) -> GraphState:
    retriever = get_dense_retriever(k=6)

    # Optional advanced retriever (Cohere rerank) if configured
    rerank = try_get_rerank_retriever(retriever, top_n=6)
    active_retriever = rerank if rerank else retriever

    docs = active_retriever.invoke(state["question"])
    state["retrieved_docs"] = docs
    state["use_web"] = should_use_web(state["question"], docs)
    state["web_snippets"] = ""
    return state

def web_search_node(state: GraphState) -> GraphState:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        state["web_snippets"] = "(Tavily not configured: set TAVILY_API_KEY to enable web search.)"
        return state

    client = TavilyClient(api_key=api_key)
    res = client.search(query=state["question"], max_results=5)
    # Convert to a compact string the LLM can use
    parts = []
    for r in res.get("results", []):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        parts.append(f"- {title}\n  {content}\n  Source: {url}")
    state["web_snippets"] = "\n".join(parts) if parts else "(No web results.)"
    return state

def answer_node(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build context with citations
    ctx_blocks = []
    for i, d in enumerate(state["retrieved_docs"][:8], start=1):
        src = d.metadata.get("source", "unknown")
        ctx_blocks.append(f"[KB {i} | {src}]\n{d.page_content}")

    kb_context = "\n\n".join(ctx_blocks) if ctx_blocks else "(No KB context retrieved.)"

    system = SystemMessage(
        content=(
            "You are LearnForge, a learning assistant. "
            "Answer using the user's KB context first. "
            "If web snippets are present, use them only to supplement. "
            "Do not fabricate. If the KB doesn't contain it, say so.\n\n"
            "Return:\n"
            "1) Answer\n"
            "2) Citations: list the KB tags you used like [KB 2 | session_07.md]\n"
        )
    )

    user = HumanMessage(
        content=(
            f"Question:\n{state['question']}\n\n"
            f"KB Context:\n{kb_context}\n\n"
            f"Web Snippets:\n{state['web_snippets']}\n"
        )
    )

    resp = llm.invoke([system, user])
    state["answer"] = resp.content
    return state

def route_after_retrieve(state: GraphState) -> str:
    return "web_search" if state["use_web"] else "answer"

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("web_search", web_search_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"web_search": "web_search", "answer": "answer"})
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    return g.compile()

GRAPH = build_graph()

def ask(question: str) -> GraphState:
    return GRAPH.invoke({"question": question, "use_web": False, "retrieved_docs": [], "web_snippets": "", "answer": ""})