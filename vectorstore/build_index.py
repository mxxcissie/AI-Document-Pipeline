from app.pipeline.document_loader import load_and_chunk_documents
from vectorstore.embedding_service import embed_texts
from vectorstore.faiss_store import FAISSStore


def build_vector_store(data_dir: str = "data/sample_docs") -> FAISSStore:
    chunks = load_and_chunk_documents(data_dir)

    if not chunks:
        raise ValueError("No documents found to build vector store")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    if not embeddings:
        raise ValueError("Failed to generate embeddings")

    dimension = len(embeddings[0])

    store = FAISSStore(dimension=dimension)
    store.add(embeddings, chunks)

    return store


vector_store = build_vector_store()