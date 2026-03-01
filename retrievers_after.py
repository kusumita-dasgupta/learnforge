import os
import glob
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from langchain_community.retrievers import BM25Retriever

KB_DIR = "kb"
QDRANT_PATH = ".qdrant"
COLLECTION = "learnforge_kb"

_client = QdrantClient(path=QDRANT_PATH)

def _load_kb_docs() -> List[Document]:
    paths = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    if not paths:
        raise FileNotFoundError(f"No markdown files found in {KB_DIR}/")
    docs: List[Document] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))
    return docs

def _chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=180)
    return splitter.split_documents(docs)

def _get_vectorstore() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

def _dense_docs(query: str, k: int) -> List[Document]:
    vs = _get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

def _bm25_docs(query: str, k: int) -> List[Document]:
    docs = _chunk_docs(_load_kb_docs())
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25.invoke(query)

def get_hybrid_retriever_after(k_dense: int = 6, k_bm25: int = 6) -> BaseRetriever:
    class HybridRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> List[Document]:
            bm25_hits = _bm25_docs(query, k_bm25)
            dense_hits = _dense_docs(query, k_dense)

            seen = set()
            merged: List[Document] = []

            def doc_id(d: Document) -> str:
                src = d.metadata.get("source", "unknown")
                return f"{src}::{hash(d.page_content[:200])}"

            for d in bm25_hits + dense_hits:
                key = doc_id(d)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)

            return merged

    return HybridRetriever()