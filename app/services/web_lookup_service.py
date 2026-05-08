import logging
from typing import Any

import requests

from app.core.config import WEB_LOOKUP_ENABLED, WEB_LOOKUP_URL

logger = logging.getLogger(__name__)


def _build_result(source: str, title: str, text: str, score: float = 0.0) -> dict[str, Any]:
    return {
        "source": source,
        "chunk_id": source,
        "text": f"{title}\n{text}".strip(),
        "score": score,
    }


def external_lookup(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    if not WEB_LOOKUP_ENABLED:
        return []

    try:
        response = requests.get(
            WEB_LOOKUP_URL,
            params={
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.exception("External lookup failed")
        return []

    results: list[dict[str, Any]] = []

    abstract = str(payload.get("AbstractText", "")).strip()
    heading = str(payload.get("Heading", "")).strip() or query
    abstract_url = str(payload.get("AbstractURL", "")).strip() or "external_lookup:abstract"

    if abstract:
        results.append(_build_result(abstract_url, heading, abstract, score=0.0))

    related_topics = payload.get("RelatedTopics", [])
    for index, topic in enumerate(related_topics):
        if len(results) >= max_results:
            break

        if not isinstance(topic, dict):
            continue

        if "Topics" in topic and isinstance(topic["Topics"], list):
            topic_items = topic["Topics"]
        else:
            topic_items = [topic]

        for item in topic_items:
            if len(results) >= max_results:
                break
            if not isinstance(item, dict):
                continue

            text = str(item.get("Text", "")).strip()
            first_url = str(item.get("FirstURL", "")).strip()
            if not text:
                continue

            results.append(
                _build_result(
                    first_url or f"external_lookup:related:{index}",
                    query,
                    text,
                    score=0.1,
                )
            )

    return results[:max_results]
