from sklearn.feature_extraction.text import TfidfVectorizer

_vectorizer = None


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer()
    return _vectorizer


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    vectorizer = get_vectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors.toarray().tolist()


def embed_query(text: str) -> list[float]:
    if not text:
        raise ValueError("Query text cannot be empty")

    vectorizer = get_vectorizer()
    vector = vectorizer.transform([text])
    return vector.toarray()[0].tolist()