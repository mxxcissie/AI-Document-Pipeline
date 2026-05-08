from typing import List, Optional

from pydantic import BaseModel, Field


class BaseQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User query")


class QueryRequest(BaseQueryRequest):
    pass


class SearchRequest(BaseQueryRequest):
    top_k: int = Field(3, ge=1, le=10, description="Number of results to retrieve")


class RAGQueryRequest(BaseQueryRequest):
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


class AgentQueryRequest(BaseQueryRequest):
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")
    max_steps: int = Field(3, ge=1, le=5, description="Maximum planner/execution steps")


class Source(BaseModel):
    source: str
    chunk_id: str
    text: str
    score: Optional[float] = None


class Metrics(BaseModel):
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    retrieved_chunks: int
    relevant_chunks: int
    cache: Optional[str] = None


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    metrics: Metrics


class AgentStep(BaseModel):
    step: int
    action: str
    query: str
    reason: str
    observation: Optional[str] = None


class AgentMetrics(BaseModel):
    planner_time_ms: float
    tool_execution_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_steps: int
    num_tool_calls: int
    cache: Optional[str] = None
    fallback_to_rag: bool = False


class AgentResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    steps: List[AgentStep]
    metrics: AgentMetrics
