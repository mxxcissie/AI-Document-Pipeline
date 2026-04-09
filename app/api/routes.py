from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/query")
def query_document(request: QueryRequest):
    return {
        "message": "Query endpoint is working",
        "question": request.question,
        "answer": "Placeholder response. LLM integration will be added on Day 2."
    }