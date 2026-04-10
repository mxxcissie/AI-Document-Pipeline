import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings: list[list[float]], docs: list[dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(docs)

    def search(self, query_embedding: list[float], top_k: int = 3):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = dict(self.documents[idx])
                doc["score"] = float(distances[0][rank])
                results.append(doc)

        return results