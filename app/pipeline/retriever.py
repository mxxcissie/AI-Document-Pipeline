from typing import Any

from app.pipeline.embedder import embed_text
from vectorstore.build_index import get_vector_store


def retrieve_documents(
    query: str,
    top_k: int = 3,
    max_distance: float = 2.0,
) -> list[dict[str, Any]]:
    vector_store = get_vector_store()

    query_embedding = embed_text(query)

    results = vector_store.search(query_embedding, top_k=top_k)

    return [
        result
        for result in results
        if result.get("score", float("inf")) <= max_distance
    ]
