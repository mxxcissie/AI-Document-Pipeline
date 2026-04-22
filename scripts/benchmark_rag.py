import os
import statistics
import subprocess
import time
from hashlib import sha256

import redis
import requests

BASE_URL = "http://127.0.0.1:8000"
QUESTION = "What is RAG?"
TOP_K = 3
HIT_RUNS = 10


def make_cache_key(question: str, top_k: int) -> str:
    raw = f"rag:{question.strip().lower()}::top_k={top_k}"
    return f"rag:{sha256(raw.encode('utf-8')).hexdigest()}"


def measure_once() -> tuple[float, str | None]:
    start = time.perf_counter()

    response = requests.post(
        f"{BASE_URL}/api/rag-query",
        json={"question": QUESTION, "top_k": TOP_K},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000.0
    cache_status = data.get("metrics", {}).get("cache")
    return elapsed_ms, cache_status


def clear_benchmark_cache() -> bool:
    cache_key = make_cache_key(QUESTION, TOP_K)
    redis_url = os.getenv("REDIS_URL")

    if redis_url:
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.delete(cache_key)
            return True
        except Exception:
            pass

    try:
        subprocess.run(
            ["docker", "exec", "ai-redis", "redis-cli", "DEL", cache_key],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def main():
    print("Running RAG cache benchmark...\n")

    cache_cleared = clear_benchmark_cache()
    if cache_cleared:
        print("Cleared benchmark cache key before running.\n")
    else:
        print("Could not clear benchmark cache key; running with observed cache state.\n")

    first_ms, first_cache = measure_once()

    hit_results = []
    hit_cache_values = []
    for _ in range(HIT_RUNS):
        elapsed_ms, cache_status = measure_once()
        hit_results.append(elapsed_ms)
        hit_cache_values.append(cache_status)

    avg_hit = statistics.mean(hit_results)
    min_hit = min(hit_results)
    max_hit = max(hit_results)

    print(f"Question: {QUESTION}")
    print(f"First request latency: {first_ms:.2f} ms")
    print(f"First request cache status: {first_cache or 'unknown'}")
    print(f"Follow-up avg latency over {HIT_RUNS} runs: {avg_hit:.2f} ms")
    print(f"Follow-up min latency: {min_hit:.2f} ms")
    print(f"Follow-up max latency: {max_hit:.2f} ms")
    print(f"Follow-up cache statuses: {sorted(set(hit_cache_values))}")

    if avg_hit > 0:
        ratio = first_ms / avg_hit
        if cache_cleared and first_cache == "miss":
            print(f"Approx speedup (miss / avg hit): {ratio:.2f}x")
        else:
            print(f"Observed first-request / avg follow-up ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()