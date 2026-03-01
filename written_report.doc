           The Certification Challenge
LearnForge: An Agentic RAG Learning Architect
Task 1: Defining Problem, Audience, and Scope
1.1 Write a succinct 1-sentence description of the problem (2 pts)
AI engineers consume large volumes of technical learning material but lack a structured, retrieval-driven system that allows them to query, evaluate, and iteratively improve their understanding using measurable feedback.

1.2 Why is this a problem for your specific user? (5 pts)
The target user is an AI engineer or architect actively studying topics such as RAG pipelines, LangGraph memory systems, evaluation frameworks like RAGAS, and multi-agent orchestration. Over weeks of study, they accumulate markdown notes, notebooks, diagrams, and technical summaries. However, this information remains fragmented and manually searchable. When preparing for interviews, building production systems, or revisiting previous learnings, they rely on keyword search or memory rather than semantic retrieval.
The deeper issue is the absence of an evaluation loop. Even if they build a RAG system over their notes, they typically do not measure context recall, faithfulness, or precision. As a result, improvements are intuitive rather than data-driven. LearnForge solves this by indexing personal learning materials, retrieving relevant context, generating grounded responses, and quantitatively evaluating system performance using RAGAS.

1.3 Create a list of evaluation questions (2 pts)
The following evaluation questions form the golden dataset:
What is KV cache in transformer decoding and why does it matter?


Explain MemorySaver vs Store in LangGraph.


When can BM25 outperform dense retrieval?


Why does chunk overlap matter in RAG systems?


What is contextual compression?


Each question includes a reference answer and is evaluated using RAGAS.

Task 2: Propose a Solution

2.1 Proposed Solution (6 pts)
LearnForge is a local Agentic RAG system that converts personal learning materials into a structured knowledge engine. The user interacts through a local endpoint. The system retrieves semantically relevant context from indexed notes, generates grounded responses using an LLM, and evaluates outputs using RAGAS.
The system was intentionally designed to support controlled experimentation. Two retrieval pipelines were implemented—baseline dense retrieval and hybrid retrieval—to enable measurable improvement analysis.

2.2 Infrastructure Diagram + Tooling Justification (7 pts)
Infrastructure Diagram
┌────────────────────────────┐
│           User             │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│      FastAPI Endpoint      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│     LangGraph Orchestration│
│  (Retrieval + Generation)  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   Hybrid Retriever         │
│  BM25  +  Dense Vector     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│      Qdrant Vector DB      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Embeddings + LLM          │
│ text-embedding-3-small     │
│ gpt-4o-mini                │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│        RAGAS Eval          │
└────────────────────────────┘




Tooling Choices
Component
Tool
Reason
LLM
gpt-4o-mini
High reasoning quality, cost efficient
Orchestration
LangGraph
Production-style state control
Retriever
Dense + BM25
Balances semantic + lexical retrieval
Embeddings
text-embedding-3-small
Strong semantic encoding
Vector DB
Qdrant
Lightweight + production compatible
Evaluation
RAGAS
Quantitative RAG benchmarking
UI
FastAPI local
Controlled demo environment


2.3 RAG and Agent Components (2 pts)
RAG components include chunking, embedding, indexing, retrieval, and generation. The agent component consists of LangGraph nodes controlling retrieval and generation flow. The system is modular and supports retrieval experimentation without altering orchestration.

Task 3: Dealing with the Data

3.1 Data Sources and External APIs (5 pts)
Primary data includes AI Makerspace session notes, LangGraph documentation excerpts, and retrieval benchmarking summaries. These documents are embedded and indexed into Qdrant.
The system is designed to support external APIs such as Tavily for web search; however, this iteration focuses on optimizing personal corpus retrieval.

3.2 Default Chunking Strategy (5 pts)
The system uses RecursiveCharacterTextSplitter with moderate chunk size and overlap. Smaller chunks increase precision but risk fragmenting context. Larger chunks preserve concept continuity but increase noise. Overlap ensures boundary continuity, especially for technical explanations spanning multiple sentences.

Task 4: Build End-to-End Prototype
A complete local prototype was built using FastAPI, LangGraph orchestration, Qdrant vector storage, OpenAI embeddings, and RAGAS evaluation harness. Two retrieval pipelines were implemented:
Baseline Dense Retrieval


Hybrid Retrieval (BM25 + Dense)


Evaluation outputs are saved as CSV files:
eval/results.csv (baseline)


eval/results_after.csv (hybrid)



Task 5: Evals

5.1 Baseline Evaluation (10 pts)
Baseline Mean Scores (Dense Retrieval):
Metric
Score
Faithfulness
0.7083
Context Precision
0.6014
Context Recall
0.6667
Answer Relevancy
0.6224

Baseline Interpretation (5 pts)
The baseline system demonstrates moderate faithfulness and precision but suboptimal recall (0.6667). This indicates missed retrieval of relevant supporting context. The low answer relevancy (0.6224) suggests that incomplete context limits response quality. The primary structural weakness is insufficient retrieval coverage.

Task 6: Improving Your Prototype
6.1 Advanced Retrieval Technique (2 pts)
I implemented Hybrid Retrieval combining BM25 and Dense Vector retrieval. Dense retrieval captures semantic similarity, while BM25 captures exact lexical matches. This is particularly useful for technical corpora containing specific class names, parameters, and framework terminology.

6.2 Implementation (10 pts)
The hybrid retriever retrieves top-k results from both BM25 and dense search, merges results, deduplicates them, and passes the combined context to the LLM. The orchestration layer remains unchanged, ensuring fair A/B comparison.

6.3 Performance Comparison (2 pts)
Hybrid Mean Scores:
Metric
Before
After
Faithfulness
0.7083
0.6667
Context Precision
0.6014
0.5821
Context Recall
0.6667
1.0000
Answer Relevancy
0.6224
0.9486

Interpretation
Context recall improved dramatically from 0.6667 to 1.0000, meaning the correct supporting chunk is always retrieved. Answer relevancy improved from 0.6224 to 0.9486, indicating significantly stronger alignment between query and response. Precision and faithfulness decreased slightly due to broader retrieval introducing minor additional noise—an expected recall-precision tradeoff. The structural gain in recall outweighs the minor reduction in precision.

Task 7: Next Steps

I will retain Hybrid Retrieval for Demo Day. Achieving 1.000 context recall eliminates retrieval failure as a structural weakness. The substantial increase in answer relevancy demonstrates meaningful system improvement. Future enhancements may include reranking, contextual compression, and latency benchmarking.

Final Submission Contents
The GitHub repository includes:
Full source code


Baseline and hybrid pipelines


Evaluation scripts


CSV results


Infrastructure diagram


5-minute Loom demo


This written report




