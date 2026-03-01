import json
import os
from dotenv import load_dotenv

import pandas as pd
from datasets import Dataset

from rag_agent_after import ask_after

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

DATA_PATH = os.path.join("eval", "golden_dataset.jsonl")
OUT_PATH = os.path.join("eval", "results_after.csv")

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    rows = load_jsonl(DATA_PATH)

    records = []
    for r in rows:
        q = r["question"]
        gt = r.get("ground_truth", "")

        out = ask_after(q)
        answer = out["answer"]
        contexts = [d.page_content for d in out.get("retrieved_docs", [])]

        records.append(
            {"question": q, "answer": answer, "contexts": contexts, "ground_truth": gt}
        )

    ds = Dataset.from_list(records)

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)

    df = result.to_pandas()
    df.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)
    print(df.describe(include="all"))

if __name__ == "__main__":
    main()