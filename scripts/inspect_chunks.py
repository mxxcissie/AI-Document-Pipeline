from app.pipeline.document_loader import load_and_chunk_documents


def preview_chunks(
    directory: str = "data/sample_docs",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
):
    chunks = load_and_chunk_documents(
        directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"\nTotal chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "-" * 40)
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Chunk Index: {chunk.get('chunk_index')}")
        print(chunk["text"])


if __name__ == "__main__":
    preview_chunks()