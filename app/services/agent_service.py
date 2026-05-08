import json
import logging
import time
from typing import Any

from app.pipeline.retriever import retrieve_documents
from app.services.cache import get_cached_agent_response, set_cached_agent_response
from app.services.llm_factory import get_llm_service
from app.services.rag_service import answer_with_rag, build_context, build_rag_prompt
from app.services.web_lookup_service import external_lookup

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_SCORE = 1.0
MAX_EMPTY_EXTERNAL_LOOKUPS = 2


class AgentPlannerError(RuntimeError):
    pass


def build_planner_prompt(question: str, context: list[dict[str, Any]]) -> str:
    context_summary = json.dumps(context, ensure_ascii=True, indent=2)

    return f"""
You are planning the next action for a lightweight RAG agent.

Available actions:
1. retrieve_documents
2. refine_query
3. external_lookup
4. generate_answer
5. stop

Rules:
- Use retrieve_documents when more context is needed.
- Use refine_query when the last retrieval was weak or too broad.
- Use external_lookup when the internal document corpus is likely insufficient or the user asks for a broader comparison.
- Use generate_answer when the retrieved context is sufficient.
- Prefer external_lookup for comparison questions involving technologies, products, or systems that are not both clearly covered in the internal document corpus.
- Prefer external_lookup when the question includes terms like compare, versus, vs, trade-offs, or pros and cons and the available context is weak.
- Return JSON only with keys: action, query, reason.
- Keep query concise and tool-focused.

User question:
{question}

Current context summary:
{context_summary}
""".strip()


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AgentPlannerError("Planner did not return a JSON object")

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AgentPlannerError("Planner returned invalid JSON") from exc

    action = str(payload.get("action", "")).strip()
    query = str(payload.get("query", "")).strip()
    reason = str(payload.get("reason", "")).strip()

    if not action or not reason:
        raise AgentPlannerError("Planner response missing required fields")

    if not query:
        query = ""

    return {
        "action": action,
        "query": query,
        "reason": reason,
    }


def _make_context_observation(
    retrieved_chunks: list[dict[str, Any]],
    *,
    tool_name: str = "retrieve_documents",
) -> dict[str, Any]:
    if not retrieved_chunks:
        return {
            "tool": tool_name,
            "retrieved_chunks": 0,
            "top_score": None,
            "low_confidence": True,
            "sources": [],
        }

    top_score = retrieved_chunks[0].get("score")
    low_confidence = top_score is None or top_score > LOW_CONFIDENCE_SCORE

    return {
        "tool": tool_name,
        "retrieved_chunks": len(retrieved_chunks),
        "top_score": top_score,
        "low_confidence": low_confidence,
        "sources": [chunk.get("source") for chunk in retrieved_chunks[:3]],
    }


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk)

    return deduped


def _generate_answer(question: str, chunks: list[dict[str, Any]]) -> tuple[str, float]:
    if not chunks:
        return "I do not have enough information from the provided documents.", 0.0

    llm = get_llm_service()
    prompt = build_rag_prompt(question, build_context(chunks))

    generation_start = time.perf_counter()
    answer = llm.generate(prompt)
    generation_time_ms = round((time.perf_counter() - generation_start) * 1000, 2)

    return answer, generation_time_ms


def answer_with_agent(question: str, top_k: int = 3, max_steps: int = 3) -> dict[str, Any]:
    total_start = time.perf_counter()

    cached = get_cached_agent_response(question, top_k, max_steps)
    if cached:
        metrics = cached.get("metrics", {})
        metrics["cache"] = "hit"
        cached["metrics"] = metrics
        return cached

    planner_time_ms = 0.0
    tool_execution_time_ms = 0.0
    retrieval_time_ms = 0.0
    generation_time_ms = 0.0
    num_tool_calls = 0
    steps: list[dict[str, Any]] = []
    context_log: list[dict[str, Any]] = []
    gathered_chunks: list[dict[str, Any]] = []
    fallback_to_rag = False
    empty_external_lookup_count = 0

    llm = get_llm_service()

    try:
        for step_number in range(1, max_steps + 1):
            planner_start = time.perf_counter()
            planner_output = llm.generate(build_planner_prompt(question, context_log))
            planner_time_ms += round((time.perf_counter() - planner_start) * 1000, 2)

            plan = _extract_json_object(planner_output)
            action = plan["action"]
            query = plan["query"] or question
            reason = plan["reason"]

            if action in {"generate_answer", "stop"}:
                answer, answer_generation_ms = _generate_answer(
                    question,
                    _dedupe_chunks(gathered_chunks),
                )
                generation_time_ms += answer_generation_ms
                result = {
                    "question": question,
                    "answer": answer,
                    "sources": _dedupe_chunks(gathered_chunks),
                    "steps": steps
                    + [
                        {
                            "step": step_number,
                            "action": action,
                            "query": query,
                            "reason": reason,
                            "observation": "Generated final answer from collected context",
                        }
                    ],
                    "metrics": {
                        "planner_time_ms": round(planner_time_ms, 2),
                        "tool_execution_time_ms": round(tool_execution_time_ms, 2),
                        "retrieval_time_ms": round(retrieval_time_ms, 2),
                        "generation_time_ms": round(generation_time_ms, 2),
                        "total_time_ms": round((time.perf_counter() - total_start) * 1000, 2),
                        "num_steps": len(steps) + 1,
                        "num_tool_calls": num_tool_calls,
                        "cache": "miss",
                        "fallback_to_rag": fallback_to_rag,
                    },
                }
                set_cached_agent_response(question, top_k, max_steps, result)
                return result

            if action not in {"retrieve_documents", "refine_query", "external_lookup"}:
                raise AgentPlannerError(f"Unsupported planner action: {action}")

            execution_start = time.perf_counter()
            if action == "external_lookup":
                retrieval_start = time.perf_counter()
                retrieved_chunks = external_lookup(query, max_results=top_k)
                retrieval_elapsed_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
            else:
                retrieval_start = time.perf_counter()
                retrieved_chunks = retrieve_documents(query, top_k=top_k)
                retrieval_elapsed_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
            retrieval_time_ms += retrieval_elapsed_ms

            tool_execution_time_ms += round((time.perf_counter() - execution_start) * 1000, 2)
            num_tool_calls += 1

            gathered_chunks.extend(retrieved_chunks)
            observation = _make_context_observation(retrieved_chunks, tool_name=action)
            context_log.append(
                {
                    "step": step_number,
                    "action": action,
                    "query": query,
                    "reason": reason,
                    **observation,
                }
            )

            observation_text = (
                f"{action} returned {observation['retrieved_chunks']} chunks; "
                f"top_score={observation['top_score']}; "
                f"low_confidence={observation['low_confidence']}"
            )

            if action == "external_lookup" and observation["retrieved_chunks"] == 0:
                empty_external_lookup_count += 1
                if empty_external_lookup_count >= MAX_EMPTY_EXTERNAL_LOOKUPS:
                    observation_text += "; stopping repeated empty external lookups"
                    steps.append(
                        {
                            "step": step_number,
                            "action": action,
                            "query": query,
                            "reason": reason,
                            "observation": observation_text,
                        }
                    )
                    break
            elif action == "external_lookup":
                empty_external_lookup_count = 0

            steps.append(
                {
                    "step": step_number,
                    "action": action,
                    "query": query,
                    "reason": reason,
                    "observation": observation_text,
                }
            )

        answer, answer_generation_ms = _generate_answer(question, _dedupe_chunks(gathered_chunks))
        generation_time_ms += answer_generation_ms
        result = {
            "question": question,
            "answer": answer,
            "sources": _dedupe_chunks(gathered_chunks),
            "steps": steps,
            "metrics": {
                "planner_time_ms": round(planner_time_ms, 2),
                "tool_execution_time_ms": round(tool_execution_time_ms, 2),
                "retrieval_time_ms": round(retrieval_time_ms, 2),
                "generation_time_ms": round(generation_time_ms, 2),
                "total_time_ms": round((time.perf_counter() - total_start) * 1000, 2),
                "num_steps": len(steps),
                "num_tool_calls": num_tool_calls,
                "cache": "miss",
                "fallback_to_rag": fallback_to_rag,
            },
        }
        set_cached_agent_response(question, top_k, max_steps, result)
        return result

    except Exception:
        logger.exception("Agent flow failed; falling back to standard RAG")
        fallback_to_rag = True
        rag_result = answer_with_rag(question, top_k=top_k)

        result = {
            "question": rag_result["question"],
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "steps": steps
            + [
                {
                    "step": len(steps) + 1,
                    "action": "fallback_to_rag",
                    "query": question,
                    "reason": "Planner or tool execution failed",
                    "observation": "Returned standard RAG response",
                }
            ],
            "metrics": {
                "planner_time_ms": round(planner_time_ms, 2),
                "tool_execution_time_ms": round(tool_execution_time_ms, 2),
                "retrieval_time_ms": rag_result["metrics"]["retrieval_time_ms"],
                "generation_time_ms": rag_result["metrics"]["generation_time_ms"],
                "total_time_ms": round((time.perf_counter() - total_start) * 1000, 2),
                "num_steps": len(steps) + 1,
                "num_tool_calls": num_tool_calls,
                "cache": rag_result["metrics"].get("cache", "miss"),
                "fallback_to_rag": fallback_to_rag,
            },
        }
        set_cached_agent_response(question, top_k, max_steps, result)
        return result
