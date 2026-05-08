from unittest.mock import Mock, patch

from app.services.agent_service import answer_with_agent
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


@patch("app.services.agent_service.set_cached_agent_response")
@patch("app.services.agent_service.get_cached_agent_response", return_value=None)
@patch("app.services.agent_service.retrieve_documents")
@patch("app.services.agent_service.get_llm_service")
def test_answer_with_agent_runs_planner_loop(
    mock_get_llm,
    mock_retrieve,
    mock_get_cached,
    mock_set_cached,
):
    mock_retrieve.return_value = [
        {
            "source": "doc1.txt",
            "chunk_id": "doc1.txt_0",
            "text": "Redis reduces latency for repeated queries.",
            "score": 0.2,
        }
    ]

    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        '{"action":"retrieve_documents","query":"redis repeated query latency","reason":"Need context first"}',
        '{"action":"generate_answer","query":"redis repeated query latency","reason":"Context is sufficient"}',
        "Redis caching reduces latency for repeated queries.",
    ]
    mock_get_llm.return_value = mock_llm

    result = answer_with_agent("Why use Redis here?", top_k=3, max_steps=3)

    assert result["question"] == "Why use Redis here?"
    assert result["answer"] == "Redis caching reduces latency for repeated queries."
    assert len(result["steps"]) == 2
    assert result["steps"][0]["action"] == "retrieve_documents"
    assert result["steps"][1]["action"] == "generate_answer"
    assert result["metrics"]["num_tool_calls"] == 1
    assert result["metrics"]["fallback_to_rag"] is False
    assert result["metrics"]["cache"] == "miss"


@patch("app.services.agent_service.set_cached_agent_response")
@patch("app.services.agent_service.get_cached_agent_response", return_value=None)
@patch("app.services.agent_service.external_lookup")
@patch("app.services.agent_service.get_llm_service")
def test_answer_with_agent_can_use_external_lookup(
    mock_get_llm,
    mock_external_lookup,
    mock_get_cached,
    mock_set_cached,
):
    mock_external_lookup.return_value = [
        {
            "source": "https://example.com/redis",
            "chunk_id": "https://example.com/redis",
            "text": "Redis is an in-memory cache, while DynamoDB is a managed database service.",
            "score": 0.0,
        }
    ]

    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        '{"action":"external_lookup","query":"Redis vs DynamoDB system tradeoffs","reason":"Need broader context than internal docs"}',
        '{"action":"generate_answer","query":"Redis vs DynamoDB system tradeoffs","reason":"External context is sufficient"}',
        "Redis is best for low-latency caching, while DynamoDB is better for durable managed storage.",
    ]
    mock_get_llm.return_value = mock_llm

    result = answer_with_agent("Compare Redis and DynamoDB for this system", top_k=3, max_steps=3)

    assert result["question"] == "Compare Redis and DynamoDB for this system"
    assert result["answer"].startswith("Redis is best for low-latency caching")
    assert result["steps"][0]["action"] == "external_lookup"
    assert result["metrics"]["num_tool_calls"] == 1
    assert result["metrics"]["fallback_to_rag"] is False


@patch("app.services.agent_service.set_cached_agent_response")
@patch("app.services.agent_service.get_cached_agent_response", return_value=None)
@patch("app.services.agent_service.external_lookup", return_value=[])
@patch("app.services.agent_service.get_llm_service")
def test_answer_with_agent_stops_repeated_empty_external_lookups(
    mock_get_llm,
    mock_external_lookup,
    mock_get_cached,
    mock_set_cached,
):
    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        '{"action":"external_lookup","query":"Redis vs DynamoDB comparison","reason":"Need broader context than internal docs"}',
        '{"action":"external_lookup","query":"Redis vs DynamoDB comparison","reason":"Try external context again"}',
    ]
    mock_get_llm.return_value = mock_llm

    result = answer_with_agent("Compare Redis and DynamoDB for this system", top_k=3, max_steps=3)

    assert len(result["steps"]) == 2
    assert result["steps"][0]["action"] == "external_lookup"
    assert result["steps"][1]["action"] == "external_lookup"
    assert "stopping repeated empty external lookups" in result["steps"][1]["observation"]
    assert result["metrics"]["num_tool_calls"] == 2
    assert result["metrics"]["fallback_to_rag"] is False


@patch("app.services.agent_service.set_cached_agent_response")
@patch("app.services.agent_service.get_cached_agent_response", return_value=None)
@patch("app.services.agent_service.answer_with_rag")
@patch("app.services.agent_service.get_llm_service")
def test_answer_with_agent_falls_back_to_rag_on_bad_planner_output(
    mock_get_llm,
    mock_answer_with_rag,
    mock_get_cached,
    mock_set_cached,
):
    mock_llm = Mock()
    mock_llm.generate.return_value = "not valid json"
    mock_get_llm.return_value = mock_llm
    mock_answer_with_rag.return_value = {
        "question": "What is RAG?",
        "answer": "Fallback answer",
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

    result = answer_with_agent("What is RAG?", top_k=3, max_steps=2)

    assert result["question"] == "What is RAG?"
    assert result["answer"] == "Fallback answer"
    assert result["metrics"]["fallback_to_rag"] is True
    assert result["steps"][-1]["action"] == "fallback_to_rag"
