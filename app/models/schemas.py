from pydantic import BaseModel, Field
from typing import List, Optional


class BaseQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User query")


class QueryRequest(BaseQueryRequest):
    pass


class SearchRequest(BaseQueryRequest):
    top_k: int = Field(3, ge=1, le=10, description="Number of results to retrieve")


class RAGQueryRequest(BaseQueryRequest):
    top_k: int = Field(3, ge=1, le=10)


class Source(BaseModel):
    content: str
    score: Optional[float] = None


class Metrics(BaseModel):
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    retrieved_chunks: int
    relevant_chunks: int


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    metrics: Metrics