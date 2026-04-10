from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class SearchRequest(BaseModel):
    question: str
    top_k: int = 3


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3