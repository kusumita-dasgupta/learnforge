import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.retrievers import BaseRetriever

QDRANT_PATH = ".qdrant"
COLLECTION = "learnforge_kb"

# ✅ Create ONE global client instance
_client = QdrantClient(path=QDRANT_PATH)

def get_vectorstore() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

def get_dense_retriever(k: int = 6) -> BaseRetriever:
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})

def try_get_rerank_retriever(base_retriever: BaseRetriever, top_n: int = 6) -> Optional[BaseRetriever]:
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        return None

    try:
        from langchain_community.document_compressors import CohereRerank
        from langchain.retrievers import ContextualCompressionRetriever

        compressor = CohereRerank(top_n=top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    except Exception:
        return None