from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, SearchRequest, RAGQueryRequest
from app.services.llm_factory import get_llm_service
from app.services.rag_service import answer_with_rag
from app.pipeline.document_loader import load_and_chunk_documents
from app.pipeline.retriever import retrieve_documents

router = APIRouter()

llm = get_llm_service()


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

    except Exception:
        raise HTTPException(status_code=500, detail="LLM query failed")


@router.get("/documents/chunks")
def preview_document_chunks():
    try:
        chunks = load_and_chunk_documents("data/sample_docs")

        return {
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load documents")


@router.post("/search")
def search_documents(request: SearchRequest):
    try:
        results = retrieve_documents(request.question, top_k=request.top_k)

        return {
            "query": request.question,
            "results": results
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/rag-query")
def rag_query(request: RAGQueryRequest):
    try:
        result = answer_with_rag(request.question, top_k=request.top_k)
        return result

    except Exception:
        raise HTTPException(status_code=500, detail="RAG query failed")