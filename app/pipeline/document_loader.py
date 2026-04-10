from pathlib import Path
from app.pipeline.text_chunker import chunk_text


def load_text_documents(directory: str) -> list[dict]:
    documents = []
    base_path = Path(directory)

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for file_path in base_path.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append(
                {
                    "source": file_path.name,
                    "content": content
                }
            )

    return documents


def load_and_chunk_documents(
    directory: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> list[dict]:
    raw_documents = load_text_documents(directory)
    chunked_documents = []

    for doc in raw_documents:
        chunks = chunk_text(
            text=doc["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for index, chunk in enumerate(chunks):
            chunked_documents.append(
                {
                    "source": doc["source"],
                    "chunk_id": index,
                    "text": chunk
                }
            )

    return chunked_documents