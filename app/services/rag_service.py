from app.vectorstore.build_index import vector_store
from app.vectorstore.embedding_service import embed_query
from app.services.llm_factory import get_llm_service

llm = get_llm_service()


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for chunk in chunks:
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        context_parts.append(
            f"[Source: {source} | Chunk: {chunk_id}]\n{text}"
        )

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


def filter_relevant_chunks(chunks: list[dict], max_distance: float = 1.5) -> list[dict]:
    for chunk in chunks:
        print("Score:", chunk.get("score"))

    return [
        chunk for chunk in chunks
        if chunk.get("score", 9999.0) <= max_distance
    ]


def answer_with_rag(question: str, top_k: int = 3) -> dict:
    query_embedding = embed_query(question)
    retrieved_chunks = vector_store.search(query_embedding, top_k=top_k)
    relevant_chunks = filter_relevant_chunks(retrieved_chunks)

    if not relevant_chunks:
        return {
            "question": question,
            "answer": "I do not have enough information from the provided documents.",
            "sources": []
        }

    context = build_context(relevant_chunks)
    prompt = build_rag_prompt(question, context)
    answer = llm.generate(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": relevant_chunks
    }