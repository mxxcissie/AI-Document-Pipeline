import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.index = faiss.IndexFlatL2(dimension)
        self.documents: list[dict] = []

    def add(self, embeddings: list[list[float]], docs: list[dict]) -> None:
        if not embeddings:
            raise ValueError("embeddings cannot be empty")
        if len(embeddings) != len(docs):
            raise ValueError("embeddings and docs must have the same length")

        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.documents.extend(docs)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self.documents:
            return []

        query_vector = np.array([query_embedding], dtype="float32")
        k = min(top_k, len(self.documents))

        distances, indices = self.index.search(query_vector, k)

        results: list[dict] = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.documents):
                continue

            doc = dict(self.documents[idx])
            doc["score"] = float(distances[0][rank])
            results.append(doc)

        return results