from unittest.mock import patch

from app.core.config import DATA_DIR
from app.pipeline.document_loader import load_and_chunk_documents
from app.pipeline.text_chunker import chunk_text
from app.pipeline.retriever import retrieve_documents


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