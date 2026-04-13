import json
import logging
from hashlib import sha256
from typing import Any, Optional

import redis

from app.core.config import REDIS_URL, RAG_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

redis_client = None

if REDIS_URL:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Redis cache connected")
    except Exception as exc:
        redis_client = None
        logger.warning("Redis unavailable; continuing without cache: %s", exc)
else:
    logger.info("REDIS_URL not set; caching disabled")


def _make_cache_key(question: str, top_k: int) -> str:
    raw = f"rag:{question.strip().lower()}::top_k={top_k}"
    digest = sha256(raw.encode("utf-8")).hexdigest()
    return f"rag:{digest}"


def get_cached_rag_response(question: str, top_k: int) -> Optional[dict[str, Any]]:
    if not redis_client:
        return None

    key = _make_cache_key(question, top_k)

    try:
        cached = redis_client.get(key)
        if not cached:
            return None
        return json.loads(cached)
    except Exception:
        logger.exception("Failed to read cached RAG response")
        return None


def set_cached_rag_response(question: str, top_k: int, response: dict[str, Any]) -> None:
    if not redis_client:
        return

    key = _make_cache_key(question, top_k)

    try:
        redis_client.setex(
            key,
            RAG_CACHE_TTL_SECONDS,
            json.dumps(response),
        )
    except Exception:
        logger.exception("Failed to write cached RAG response")