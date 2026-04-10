from app.pipeline.document_loader import load_and_chunk_documents
from app.vectorstore.embedding_service import embed_texts
from app.vectorstore.faiss_store import FAISSStore

chunks = load_and_chunk_documents("data/sample_docs")

texts = [c["text"] for c in chunks]

embeddings = embed_texts(texts)

store = FAISSStore(dimension=len(embeddings[0]))
store.add(embeddings, chunks)

vector_store = store