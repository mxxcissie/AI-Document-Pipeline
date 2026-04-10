from app.pipeline.document_loader import load_and_chunk_documents

chunks = load_and_chunk_documents("data/sample_docs", chunk_size=200, chunk_overlap=40)

print(f"Total chunks: {len(chunks)}")
for chunk in chunks:
    print("-" * 40)
    print(f"Source: {chunk['source']}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(chunk["text"])