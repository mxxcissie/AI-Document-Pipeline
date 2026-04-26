from __future__ import annotations

import json
from pathlib import Path

from app.core.config import (
    DATA_DIR,
    EMBEDDING_PROVIDER,
    INDEX_DIR,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL_PATH,
    LOCAL_EMBEDDING_OFFLINE_ONLY,
)
from app.pipeline.document_loader import load_and_chunk_documents
from vectorstore.embedding_service import prepare_tfidf_vectorizer, reset_vectorizer, vectorize_texts

_vector_store = None


class StaleIndexError(RuntimeError):
    pass


def _index_file_prefix(data_dir: str) -> str:
    data_name = Path(data_dir).name
    provider = EMBEDDING_PROVIDER
    model_tag = ""

    if EMBEDDING_PROVIDER == "local_dense":
        model_source = LOCAL_EMBEDDING_MODEL_PATH or LOCAL_EMBEDDING_MODEL
        model_tag = f"_{Path(model_source).name}"

    return f"{data_name}_{provider}{model_tag}"


def get_index_paths(data_dir: str = str(DATA_DIR)) -> tuple[Path, Path]:
    prefix = _index_file_prefix(data_dir)
    return (
        INDEX_DIR / f"{prefix}.faiss",
        INDEX_DIR / f"{prefix}.json",
    )


def get_manifest_path(data_dir: str = str(DATA_DIR)) -> Path:
    prefix = _index_file_prefix(data_dir)
    return INDEX_DIR / f"{prefix}.manifest.json"


def _build_index_manifest(data_dir: str) -> dict:
    data_path = Path(data_dir)
    files = []

    for file_path in sorted(data_path.glob("*.txt")):
        stat = file_path.stat()
        files.append(
            {
                "name": file_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    manifest = {
        "data_dir": data_path.name,
        "embedding_provider": EMBEDDING_PROVIDER,
        "files": files,
    }

    if EMBEDDING_PROVIDER == "local_dense":
        manifest["embedding_model"] = LOCAL_EMBEDDING_MODEL
        if LOCAL_EMBEDDING_MODEL_PATH:
            manifest["embedding_model_path"] = LOCAL_EMBEDDING_MODEL_PATH
        manifest["embedding_offline_only"] = LOCAL_EMBEDDING_OFFLINE_ONLY

    return manifest


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _is_index_stale(data_dir: str, manifest: dict) -> bool:
    current_manifest = _build_index_manifest(data_dir)
    return manifest != current_manifest


def build_vector_store(data_dir: str = str(DATA_DIR)) -> FAISSStore:
    chunks = load_and_chunk_documents(data_dir)

    if not chunks:
        raise ValueError("No documents found to build vector store")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = vectorize_texts(texts)

    if not embeddings:
        raise ValueError("Failed to generate embeddings")

    from vectorstore.faiss_store import FAISSStore

    store = FAISSStore(dimension=len(embeddings[0]))
    store.add(embeddings, chunks)

    return store


def load_vector_store(data_dir: str = str(DATA_DIR)) -> FAISSStore:
    from vectorstore.faiss_store import FAISSStore

    index_path, metadata_path = get_index_paths(data_dir)
    manifest_path = get_manifest_path(data_dir)
    manifest = _load_manifest(manifest_path)

    if _is_index_stale(data_dir, manifest):
        raise StaleIndexError("Persisted index is stale and needs to be rebuilt")

    store = FAISSStore.load(index_path, metadata_path)

    if EMBEDDING_PROVIDER == "tfidf":
        texts = [document["text"] for document in store.documents]
        prepare_tfidf_vectorizer(texts)

    return store


def save_vector_store(store: FAISSStore, data_dir: str = str(DATA_DIR)) -> tuple[Path, Path, Path]:
    index_path, metadata_path = get_index_paths(data_dir)
    manifest_path = get_manifest_path(data_dir)
    store.save(index_path, metadata_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_build_index_manifest(data_dir), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return index_path, metadata_path, manifest_path


def reset_vector_store() -> None:
    global _vector_store
    _vector_store = None
    reset_vectorizer()


def rebuild_vector_store(data_dir: str = str(DATA_DIR), persist: bool = True) -> FAISSStore:
    global _vector_store

    reset_vector_store()
    _vector_store = build_vector_store(data_dir)

    if persist:
        save_vector_store(_vector_store, data_dir)

    return _vector_store


def get_vector_store() -> FAISSStore:
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = load_vector_store()
        except (FileNotFoundError, StaleIndexError):
            _vector_store = rebuild_vector_store()
    return _vector_store