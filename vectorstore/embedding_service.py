from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import (
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL_PATH,
    LOCAL_EMBEDDING_OFFLINE_ONLY,
)

_vectorizer = None
_dense_model = None


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer()
    return _vectorizer


def reset_vectorizer() -> None:
    global _vectorizer
    _vectorizer = None


def prepare_tfidf_vectorizer(texts: list[str]) -> None:
    if not texts:
        raise ValueError("texts cannot be empty when preparing TF-IDF vectorizer")

    vectorizer = get_vectorizer()
    vectorizer.fit(texts)


def get_local_dense_model_source() -> str:
    return LOCAL_EMBEDDING_MODEL_PATH or LOCAL_EMBEDDING_MODEL


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING_PROVIDER=local_dense"
            ) from exc

        model_source = get_local_dense_model_source()

        try:
            _dense_model = SentenceTransformer(
                model_source,
                local_files_only=LOCAL_EMBEDDING_OFFLINE_ONLY,
            )
        except Exception as exc:
            if LOCAL_EMBEDDING_OFFLINE_ONLY:
                raise RuntimeError(
                    "Failed to load the local dense model in offline-only mode. "
                    "Set LOCAL_EMBEDDING_MODEL_PATH to a downloaded model directory "
                    "or disable LOCAL_EMBEDDING_OFFLINE_ONLY."
                ) from exc

            raise RuntimeError(
                "Failed to load the local dense model. "
                "If you want fully offline rebuilds, set LOCAL_EMBEDDING_MODEL_PATH "
                "to a downloaded model directory and enable LOCAL_EMBEDDING_OFFLINE_ONLY=true."
            ) from exc
    return _dense_model


def _tfidf_vectorize_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    vectorizer = get_vectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors.toarray().tolist()


def _tfidf_vectorize_query(text: str) -> list[float]:
    if not text:
        raise ValueError("Query text cannot be empty")

    vectorizer = get_vectorizer()
    vector = vectorizer.transform([text])
    return vector.toarray()[0].tolist()


def _local_dense_vectorize_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _get_dense_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def _local_dense_vectorize_query(text: str) -> list[float]:
    if not text:
        raise ValueError("Query text cannot be empty")

    model = _get_dense_model()
    return model.encode([text], normalize_embeddings=True)[0].tolist()


def vectorize_texts(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_PROVIDER == "tfidf":
        return _tfidf_vectorize_texts(texts)
    if EMBEDDING_PROVIDER == "local_dense":
        return _local_dense_vectorize_texts(texts)

    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def vectorize_query(text: str) -> list[float]:
    if EMBEDDING_PROVIDER == "tfidf":
        return _tfidf_vectorize_query(text)
    if EMBEDDING_PROVIDER == "local_dense":
        return _local_dense_vectorize_query(text)

    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    return vectorize_texts(texts)


def embed_query(text: str) -> list[float]:
    return vectorize_query(text)