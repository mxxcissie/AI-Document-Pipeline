from typing import Any

from app.pipeline.embedder import embed_text
from vectorstore.build_index import vector_store


def retrieve_documents(
    query: str,
    top_k: int = 3,
    max_distance: float = 1.5,
) -> list[dict[str, Any]]:
    query_embedding = embed_text(query)
    results = vector_store.search(query_embedding, top_k=top_k)

    filtered_results = [
        result for result in results
        if result.get("score", 9999.0) <= max_distance
    ]

    return filtered_results