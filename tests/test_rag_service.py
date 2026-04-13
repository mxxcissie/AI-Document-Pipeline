from app.services.rag_service import answer_with_rag


def test_answer_with_rag_returns_expected_shape():
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
    assert metrics["cache"] in {"hit", "miss"}


def test_answer_with_rag_repeated_query_returns_valid_response():
    first = answer_with_rag("What is RAG?", top_k=3)
    second = answer_with_rag("What is RAG?", top_k=3)

    assert first["question"] == "What is RAG?"
    assert second["question"] == "What is RAG?"

    assert isinstance(second["answer"], str)
    assert isinstance(second["sources"], list)
    assert "metrics" in second
    assert second["metrics"]["cache"] in {"hit", "miss"}