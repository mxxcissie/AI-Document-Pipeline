import time

from app.pipeline.retriever import retrieve_documents
from app.services.llm_factory import get_llm_service

llm = get_llm_service()


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for chunk in chunks:
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        context_parts.append(f"[Source: {source} | Chunk: {chunk_id}]\n{text}")

    return "\n\n".join(context_parts)


def build_rag_prompt(question: str, context: str) -> str:
    return f"""
You are a question-answering assistant.

Rules:
- Use only the provided context.
- Do not use prior knowledge.
- Do not guess.
- If the answer is not in the context, respond exactly with:
I do not have enough information from the provided documents.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def answer_with_rag(question: str, top_k: int = 3) -> dict:
    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    retrieved_chunks = retrieve_documents(question, top_k=top_k)
    retrieval_end = time.perf_counter()
    retrieval_time_ms = round((retrieval_end - retrieval_start) * 1000, 2)

    if not retrieved_chunks:
        total_end = time.perf_counter()
        total_time_ms = round((total_end - total_start) * 1000, 2)

        print(
            f"[RAG] question='{question}' | retrieval={retrieval_time_ms}ms "
            f"| generation=0.0ms | total={total_time_ms}ms "
            f"| retrieved=0 | relevant=0"
        )

        return {
            "question": question,
            "answer": "I do not have enough information from the provided documents.",
            "sources": [],
            "metrics": {
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": 0.0,
                "total_time_ms": total_time_ms,
                "retrieved_chunks": 0,
                "relevant_chunks": 0,
            },
        }

    context = build_context(retrieved_chunks)
    prompt = build_rag_prompt(question, context)

    generation_start = time.perf_counter()
    answer = llm.generate(prompt)
    generation_end = time.perf_counter()
    generation_time_ms = round((generation_end - generation_start) * 1000, 2)

    total_end = time.perf_counter()
    total_time_ms = round((total_end - total_start) * 1000, 2)

    print(
        f"[RAG] question='{question}' | retrieval={retrieval_time_ms}ms "
        f"| generation={generation_time_ms}ms | total={total_time_ms}ms "
        f"| retrieved={len(retrieved_chunks)} | relevant={len(retrieved_chunks)}"
    )

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_chunks,
        "metrics": {
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms,
            "total_time_ms": total_time_ms,
            "retrieved_chunks": len(retrieved_chunks),
            "relevant_chunks": len(retrieved_chunks),
        },
    }