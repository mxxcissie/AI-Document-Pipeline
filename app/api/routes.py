from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_factory import get_llm_service

router = APIRouter()

llm = get_llm_service()


class QueryRequest(BaseModel):
    question: str


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/query")
def query_document(request: QueryRequest):
    try:
        prompt = f"Answer clearly:\n\n{request.question}"
        answer = llm.generate(prompt)

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))