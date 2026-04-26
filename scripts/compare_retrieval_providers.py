import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROVIDERS = ("tfidf", "local_dense")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CASES_PATH = PROJECT_ROOT / "scripts" / "retrieval_eval_cases.json"


def load_test_cases() -> list[dict]:
    return json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))


def run_provider_subprocess(provider: str) -> dict:
    env = dict(os.environ)
    env["EMBEDDING_PROVIDER"] = provider

    result = subprocess.run(
        [sys.executable, __file__, "--provider", provider, "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        return {
            "provider": provider,
            "status": "error",
            "message": result.stderr.strip() or result.stdout.strip() or "unknown error",
        }

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "provider": provider,
            "status": "error",
            "message": "provider subprocess produced no output",
        }

    return json.loads(lines[-1])


def _run_single_provider(provider: str) -> dict:
    os.environ["EMBEDDING_PROVIDER"] = provider
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    test_cases = load_test_cases()

    from vectorstore.build_index import rebuild_vector_store, reset_vector_store

    build_start = time.perf_counter()
    store = rebuild_vector_store(persist=False)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    from app.pipeline.retriever import retrieve_documents

    query_latencies_ms = []
    hits_at_1 = 0
    results_summary = []

    for case in test_cases:
        start = time.perf_counter()
        results = retrieve_documents(case["question"], top_k=3)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        query_latencies_ms.append(elapsed_ms)

        top_source = results[0]["source"] if results else None
        is_hit = top_source == case["expected_source"]
        if is_hit:
            hits_at_1 += 1

        results_summary.append(
            {
                "question": case["question"],
                "expected_source": case["expected_source"],
                "top_source": top_source,
                "hit_at_1": is_hit,
                "result_count": len(results),
                "latency_ms": round(elapsed_ms, 2),
            }
        )

    reset_vector_store()

    return {
        "provider": provider,
        "status": "ok",
        "build_ms": round(build_ms, 2),
        "avg_query_ms": round(statistics.mean(query_latencies_ms), 2),
        "min_query_ms": round(min(query_latencies_ms), 2),
        "max_query_ms": round(max(query_latencies_ms), 2),
        "hit_at_1": hits_at_1,
        "case_count": len(test_cases),
        "indexed_chunks": len(store.documents),
        "results": results_summary,
    }


def print_report(results: list[dict]) -> None:
    print("Retrieval provider comparison\n")

    for result in results:
        print(f"Provider: {result['provider']}")

        if result["status"] != "ok":
            print(f"  Status: {result['status']}")
            print(f"  Message: {result.get('message', 'unavailable')}")
            print()
            continue

        print(f"  Indexed chunks: {result['indexed_chunks']}")
        print(f"  Build time: {result['build_ms']} ms")
        print(
            f"  Query latency avg/min/max: "
            f"{result['avg_query_ms']} / {result['min_query_ms']} / {result['max_query_ms']} ms"
        )
        print(f"  Hit@1: {result['hit_at_1']} / {result['case_count']}")

        for case in result["results"]:
            print(
                f"    - {case['question']} -> "
                f"top={case['top_source']} expected={case['expected_source']} "
                f"hit={case['hit_at_1']} latency={case['latency_ms']} ms"
            )
        print()


def main() -> None:
    if "--provider" in sys.argv:
        provider = sys.argv[sys.argv.index("--provider") + 1]
        try:
            result = _run_single_provider(provider)
        except Exception as exc:
            result = {
                "provider": provider,
                "status": "skipped",
                "message": str(exc),
            }

        if "--json" in sys.argv:
            print(json.dumps(result))
        else:
            print_report([result])
        return

    results = [run_provider_subprocess(provider) for provider in PROVIDERS]
    print_report(results)


if __name__ == "__main__":
    main()