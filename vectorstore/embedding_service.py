from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = _model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    if not text:
        raise ValueError("Query text cannot be empty")

    embedding = _model.encode([text], normalize_embeddings=True)[0]
    return embedding.tolist()