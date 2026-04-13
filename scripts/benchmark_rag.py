import statistics
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
QUESTION = "What is RAG?"
TOP_K = 3
HIT_RUNS = 10


def measure_once() -> float:
    start = time.perf_counter()

    response = requests.post(
        f"{BASE_URL}/api/rag-query",
        json={"question": QUESTION, "top_k": TOP_K},
        timeout=60,
    )
    response.raise_for_status()
    _ = response.json()

    end = time.perf_counter()
    return (end - start) * 1000.0


def main():
    print("Running RAG cache benchmark...\n")

    # First run → cache miss
    miss_ms = measure_once()

    # Next runs → cache hits
    hit_results = []
    for _ in range(HIT_RUNS):
        hit_results.append(measure_once())

    avg_hit = statistics.mean(hit_results)
    min_hit = min(hit_results)
    max_hit = max(hit_results)

    print(f"Question: {QUESTION}")
    print(f"Cache miss latency: {miss_ms:.2f} ms")
    print(f"Cache hit avg latency over {HIT_RUNS} runs: {avg_hit:.2f} ms")
    print(f"Cache hit min latency: {min_hit:.2f} ms")
    print(f"Cache hit max latency: {max_hit:.2f} ms")

    if avg_hit > 0:
        print(f"Approx speedup (miss / avg hit): {miss_ms / avg_hit:.2f}x")


if __name__ == "__main__":
    main()