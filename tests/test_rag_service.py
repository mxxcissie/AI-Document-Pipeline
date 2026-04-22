from unittest.mock import Mock, patch

from app.services.rag_service import answer_with_rag


@patch("app.services.rag_service.set_cached_rag_response")
@patch("app.services.rag_service.get_cached_rag_response", return_value=None)
@patch("app.services.rag_service.get_llm_service")
@patch("app.services.rag_service.retrieve_documents")
def test_answer_with_rag_returns_expected_shape(
    mock_retrieve,
    mock_get_llm,
    mock_get_cached,
    mock_set_cached,
):
    mock_retrieve.return_value = [
        {
            "source": "doc1.txt",
            "chunk_id": "doc1.txt_0",
            "text": "RAG combines retrieval with generation.",
            "score": 0.2,
        }
    ]

    mock_llm = Mock()
    mock_llm.generate.return_value = "RAG combines retrieval with generation."
    mock_get_llm.return_value = mock_llm

    result = answer_with_rag("What is RAG?", top_k=3)

    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert "metrics" in result

    assert result["question"] == "What is RAG?"
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)

    metrics = result["metrics"]
    assert isinstance(metrics, dict)

    assert "retrieval_time_ms" in metrics
    assert "generation_time_ms" in metrics
    assert "total_time_ms" in metrics
    assert "retrieved_chunks" in metrics
    assert "relevant_chunks" in metrics
    assert "cache" in metrics

    assert isinstance(metrics["retrieval_time_ms"], (int, float))
    assert isinstance(metrics["generation_time_ms"], (int, float))
    assert isinstance(metrics["total_time_ms"], (int, float))
    assert isinstance(metrics["retrieved_chunks"], int)
    assert isinstance(metrics["relevant_chunks"], int)
    assert metrics["cache"] == "miss"


@patch("app.services.rag_service.set_cached_rag_response")
@patch("app.services.rag_service.get_cached_rag_response", return_value=None)
@patch("app.services.rag_service.retrieve_documents", return_value=[])
def test_answer_with_rag_handles_no_results(
    mock_retrieve,
    mock_get_cached,
    mock_set_cached,
):
    result = answer_with_rag("Unknown question", top_k=3)

    assert result["question"] == "Unknown question"
    assert result["answer"] == "I do not have enough information from the provided documents."
    assert result["sources"] == []
    assert result["metrics"]["generation_time_ms"] == 0.0
    assert result["metrics"]["retrieved_chunks"] == 0
    assert result["metrics"]["relevant_chunks"] == 0
    assert result["metrics"]["cache"] == "miss"


@patch("app.services.rag_service.get_cached_rag_response")
def test_answer_with_rag_cache_hit_sets_cache_flag(mock_get_cached):
    mock_get_cached.return_value = {
        "question": "What is RAG?",
        "answer": "Cached answer",
        "sources": [],
        "metrics": {
            "retrieval_time_ms": 1.0,
            "generation_time_ms": 2.0,
            "total_time_ms": 3.0,
            "retrieved_chunks": 0,
            "relevant_chunks": 0,
            "cache": "miss",
        },
    }

    result = answer_with_rag("What is RAG?", top_k=3)

    assert result["question"] == "What is RAG?"
    assert result["answer"] == "Cached answer"
    assert result["metrics"]["cache"] == "hit"
