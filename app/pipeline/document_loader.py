from pathlib import Path
from typing import Any

from app.pipeline.text_chunker import chunk_text


def load_text_documents(directory: str, pattern: str = "*.txt") -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    base_path = Path(directory)

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    for file_path in sorted(base_path.glob(pattern)):
        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append(
                {
                    "source": file_path.name,
                    "content": content,
                }
            )

    return documents


def load_and_chunk_documents(
    directory: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    raw_documents = load_text_documents(directory)
    chunked_documents: list[dict[str, Any]] = []

    for doc in raw_documents:
        chunks = chunk_text(
            text=doc["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk in enumerate(chunks):
            chunked_documents.append(
                {
                    "source": doc["source"],
                    "chunk_id": f'{doc["source"]}_{index}',
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    return chunked_documents