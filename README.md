# LearnForge — Agentic RAG Knowledge Architect

A minimal Agentic RAG app that indexes your personal learning materials (AI Makerspace Sessions 1–11) and answers questions with citations. It optionally uses Tavily web search when needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in OPENAI_API_KEY and (optional) TAVILY_API_KEY