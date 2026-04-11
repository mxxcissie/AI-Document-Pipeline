import requests
import time

BASE_URL = "http://127.0.0.1:8000/api/rag-query"

test_cases = [
    {"q": "What is RAG?", "type": "in-scope"},
    {"q": "What does a vector database store?", "type": "in-scope"},
    {"q": "Who won the 2020 NBA Finals?", "type": "out-of-scope"},
]

PASS_COUNT = 0

for case in test_cases:
    start = time.perf_counter()

    try:
        response = requests.post(
            BASE_URL,
            json={"question": case["q"], "top_k": 3},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        continue

    end = time.perf_counter()
    latency_ms = round((end - start) * 1000, 2)

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    if case["type"] == "in-scope":
        passed = bool(answer.strip()) and len(sources) > 0
    else:
        passed = "do not have enough information" in answer.lower()

    if passed:
        PASS_COUNT += 1

    print("\n====================")
    print("Question:", case["q"])
    print("Expected:", case["type"])
    print("Latency:", latency_ms, "ms")
    print("Answer:", answer)
    print("Sources:", len(sources))
    print("Result:", "✅ PASS" if passed else "❌ FAIL")

print("\n====================")
print(f"Passed {PASS_COUNT} / {len(test_cases)} tests")