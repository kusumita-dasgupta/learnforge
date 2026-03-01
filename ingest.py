import os
import glob
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

KB_DIR = "kb"
QDRANT_PATH = ".qdrant"
COLLECTION = "learnforge_kb"

def load_kb_documents() -> list[Document]:
    paths = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    if not paths:
        raise FileNotFoundError(
            f"No markdown files found in {KB_DIR}/. Create kb/session_01.md ... kb/session_11.md"
        )

    docs: list[Document] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(p)
        # Helpful metadata for citations and filtering later
        docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,     # good baseline for technical notes
        chunk_overlap=180
    )
    return splitter.split_documents(docs)

from qdrant_client.http.models import VectorParams, Distance

def build_vector_store(chunks: list[Document]) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(path=QDRANT_PATH)

    # Get embedding dimension
    dim = len(embeddings.embed_query("dimension check"))

    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    # Create collection manually
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Create vector store instance
    vs = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    # Add documents
    vs.add_documents(chunks)

    return vs

def main():
    load_dotenv()
    print("Loading KB docs...")
    docs = load_kb_documents()
    print(f"Loaded {len(docs)} files.")
    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print("Embedding + storing in Qdrant...")
    _ = build_vector_store(chunks)
    print("Done. Qdrant local DB at .qdrant/ and collection:", COLLECTION)

if __name__ == "__main__":
    main()