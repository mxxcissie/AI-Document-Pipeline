from vectorstore.embedding_service import vectorize_query


def embed_text(text: str) -> list[float]:
    return vectorize_query(text)


def vectorize_text(text: str) -> list[float]:
    return vectorize_query(text)