# LearnForge — Agentic RAG Knowledge Architect

A minimal Agentic RAG app that indexes your personal learning materials (AI Makerspace Sessions 1–11) and answers questions with citations. It optionally uses Tavily web search when needed.

## Features

- **LangGraph-based agent**: Retrieve → (optional web search) → Answer
- **Knowledge base**: Indexes markdown files from `kb/` (Sessions 1–11)
- **Vector store**: Qdrant (local `.qdrant/`) with OpenAI embeddings
- **Chunking**: RecursiveCharacterTextSplitter (chunk_size=1100, overlap=180)
- **Optional web search**: Tavily for queries about "latest", "current", "today", etc.

## Project Structure

```
learnforge/
├── kb/                    # Knowledge base (session_01.md … session_11.md)
├── rag_agent.py           # Main agent (dense + optional Cohere rerank)
├── rag_agent_after.py     # Alternative agent (hybrid BM25 + dense)
├── retrievers.py          # Dense retriever + optional Cohere rerank
├── retrievers_after.py    # Hybrid retriever (BM25 + dense, merged)
├── ingest.py              # Index KB into Qdrant
├── app.py                 # Streamlit UI
├── main.py
├── eval/
│   ├── golden_dataset.jsonl   # Evaluation questions + ground truth
│   ├── run_ragas.py           # Run RAGAS for main agent
│   ├── run_ragas_after.py     # Run RAGAS for hybrid agent
│   ├── results.csv            # Before (dense + rerank) results
│   └── results_after.csv      # After (hybrid) results
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY and (optional) TAVILY_API_KEY, COHERE_API_KEY
```

## Usage

### Index the knowledge base

```bash
python ingest.py
```

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run evaluation (RAGAS)

```bash
# Evaluate main agent (dense + optional Cohere rerank)
python -m eval.run_ragas

# Evaluate hybrid agent (BM25 + dense)
python -m eval.run_ragas_after
```

## Retrieval Configurations

| Config | Retriever | Description |
|--------|-----------|-------------|
| **Before** | Dense + optional Cohere rerank | Vector search (k=6), optionally reranked with Cohere |
| **After** | Hybrid (BM25 + Dense) | BM25 (k=6) + Dense (k=6), merged with deduplication |

## Evaluation Results (Before vs After)

RAGAS metrics on the golden dataset (3 questions):

| Metric | Before (Dense + Rerank) | After (Hybrid BM25+Dense) |
|--------|-------------------------|---------------------------|
| **Faithfulness** | 0.71 | 0.67 |
| **Context Precision** | 0.60 | 0.58 |
| **Context Recall** | 0.67 | **1.00** |
| **Answer Relevancy** | 0.62 | **0.95** |

### Per-question summary

| Question | Before | After |
|----------|--------|-------|
| **KV cache in transformer decoding** | KB doesn't contain it; low recall (0), low relevancy (0) | Improved recall (1.0) and relevancy (0.95); still out-of-KB |
| **MemorySaver vs Store in LangGraph** | Strong (faithfulness 1.0, recall 1.0, relevancy 0.93) | Similar; slight precision drop (0.80→0.75) |
| **When can BM25 outperform dense retrieval?** | Near perfect | Same; relevancy improved (0.94→0.97) |

### Takeaways

- **Hybrid retrieval** improves **context recall** (1.0 vs 0.67) and **answer relevancy** (0.95 vs 0.62).
- For in-KB questions (MemorySaver, BM25), both configs perform well; hybrid slightly boosts answer relevancy.
- For out-of-KB questions (KV cache), hybrid retrieves more context and yields more relevant answers, though faithfulness remains low when the KB lacks the topic.
