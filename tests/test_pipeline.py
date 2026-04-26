import json
from unittest.mock import patch

from app.core.config import DATA_DIR
from app.pipeline.document_loader import load_and_chunk_documents
from app.pipeline.text_chunker import chunk_text
from app.pipeline.retriever import retrieve_documents
from vectorstore.build_index import StaleIndexError, load_vector_store, rebuild_vector_store
from vectorstore.embedding_service import embed_query, get_local_dense_model_source, reset_vectorizer
from vectorstore.faiss_store import FAISSStore


def test_chunk_text_returns_chunks():
    text = "This is a simple test document. " * 20
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunk_text_invalid_arguments():
    text = "sample text"

    try:
        chunk_text(text, chunk_size=0, chunk_overlap=10)
        assert False, "Expected ValueError for chunk_size=0"
    except ValueError:
        pass

    try:
        chunk_text(text, chunk_size=50, chunk_overlap=50)
        assert False, "Expected ValueError for chunk_overlap >= chunk_size"
    except ValueError:
        pass


def test_load_and_chunk_documents_runs():
    chunks = load_and_chunk_documents(str(DATA_DIR))

    assert len(chunks) > 0
    assert all("source" in chunk for chunk in chunks)
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all(isinstance(chunk["text"], str) and chunk["text"].strip() for chunk in chunks)


def test_get_local_dense_model_source_prefers_explicit_path():
    with patch("vectorstore.embedding_service.LOCAL_EMBEDDING_MODEL_PATH", "/tmp/minilm-model"):
        with patch("vectorstore.embedding_service.LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"):
            assert get_local_dense_model_source() == "/tmp/minilm-model"


def test_faiss_store_can_save_and_load(tmp_path):
    store = FAISSStore(dimension=3)
    store.add(
        embeddings=[[0.1, 0.2, 0.3]],
        docs=[{"source": "doc1.txt", "chunk_id": "doc1.txt_0", "text": "Example result"}],
    )

    index_path = tmp_path / "test.faiss"
    metadata_path = tmp_path / "test.json"
    store.save(index_path, metadata_path)

    loaded = FAISSStore.load(index_path, metadata_path)
    results = loaded.search([0.1, 0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0]["source"] == "doc1.txt"
    assert results[0]["chunk_id"] == "doc1.txt_0"


@patch("vectorstore.build_index.get_index_paths")
def test_load_vector_store_prepares_tfidf_query_vectorizer(mock_get_index_paths, tmp_path):
    store = FAISSStore(dimension=3)
    documents = [
        {"source": "doc1.txt", "chunk_id": "doc1.txt_0", "text": "retrieval augmented generation"},
        {"source": "doc2.txt", "chunk_id": "doc2.txt_0", "text": "vector search with faiss"},
    ]
    store.add(
        embeddings=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
        docs=documents,
    )

    index_path = tmp_path / "docs.faiss"
    metadata_path = tmp_path / "docs.json"
    store.save(index_path, metadata_path)
    mock_get_index_paths.return_value = (index_path, metadata_path)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc1.txt").write_text("retrieval augmented generation", encoding="utf-8")
    (docs_dir / "doc2.txt").write_text("vector search with faiss", encoding="utf-8")
    manifest_path = tmp_path / "docs.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_dir": "docs",
                "embedding_provider": "tfidf",
                "files": [
                    {
                        "name": "doc1.txt",
                        "size": (docs_dir / "doc1.txt").stat().st_size,
                        "mtime_ns": (docs_dir / "doc1.txt").stat().st_mtime_ns,
                    },
                    {
                        "name": "doc2.txt",
                        "size": (docs_dir / "doc2.txt").stat().st_size,
                        "mtime_ns": (docs_dir / "doc2.txt").stat().st_mtime_ns,
                    },
                ],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    reset_vectorizer()
    with patch("vectorstore.build_index.get_manifest_path", return_value=manifest_path):
        loaded = load_vector_store(str(docs_dir))
        query_vector = embed_query("retrieval generation")

    assert len(loaded.documents) == 2
    assert len(query_vector) > 0


@patch("vectorstore.build_index.get_index_paths")
def test_rebuild_vector_store_persists_index(mock_get_index_paths, tmp_path):
    mock_get_index_paths.return_value = (tmp_path / "sample.faiss", tmp_path / "sample.json")
    with patch("vectorstore.build_index.get_manifest_path", return_value=tmp_path / "sample.manifest.json"):
        store = rebuild_vector_store(data_dir=str(DATA_DIR), persist=True)

    faiss_files = list(tmp_path.glob("*.faiss"))
    metadata_files = [path for path in tmp_path.glob("*.json") if not path.name.endswith(".manifest.json")]
    manifest_files = list(tmp_path.glob("*.manifest.json"))

    assert store.documents
    assert len(faiss_files) == 1
    assert len(metadata_files) == 1
    assert len(manifest_files) == 1


@patch("vectorstore.build_index.get_index_paths")
def test_load_vector_store_raises_when_manifest_is_stale(mock_get_index_paths, tmp_path):
    store = FAISSStore(dimension=3)
    store.add(
        embeddings=[[0.1, 0.2, 0.3]],
        docs=[{"source": "doc1.txt", "chunk_id": "doc1.txt_0", "text": "retrieval augmented generation"}],
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc1.txt").write_text("retrieval augmented generation", encoding="utf-8")

    index_path = tmp_path / "docs.faiss"
    metadata_path = tmp_path / "docs.json"
    manifest_path = tmp_path / "docs.manifest.json"
    store.save(index_path, metadata_path)
    manifest_path.write_text(
        json.dumps(
            {
                "data_dir": "docs",
                "embedding_provider": "tfidf",
                "files": [{"name": "doc1.txt", "size": 1, "mtime_ns": 1}],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    mock_get_index_paths.return_value = (index_path, metadata_path)

    with patch("vectorstore.build_index.get_manifest_path", return_value=manifest_path):
        try:
            load_vector_store(str(docs_dir))
            assert False, "Expected StaleIndexError for stale manifest"
        except StaleIndexError:
            pass


@patch("app.pipeline.retriever.get_vector_store")
@patch("app.pipeline.retriever.embed_text")
def test_retriever_returns_results(mock_embed_text, mock_get_vector_store):
    mock_embed_text.return_value = [0.1, 0.2, 0.3]

    mock_store = mock_get_vector_store.return_value
    mock_store.search.return_value = [
        {
            "source": "doc1.txt",
            "chunk_id": "doc1.txt_0",
            "text": "Example result",
            "score": 0.4,
        }
    ]

    results = retrieve_documents("test query", top_k=2)

    assert isinstance(results, list)
    assert len(results) == 1