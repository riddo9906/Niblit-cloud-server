#!/usr/bin/env python3
"""Stress test for llama-cpp-python inference stability.

Executes sequential and concurrent inference requests to reproduce
the llama_decode returned -1 crash deterministically.

Usage:
    python -m app.debug.stress_test [--concurrent] [--count N]
"""

import argparse
import logging
import os
import sys
import threading
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("stress_test")


def make_request(client, prompt: str, request_id: str) -> tuple[str, int, str]:
    """Send a single chat completion request. Returns (request_id, status_code, error)."""
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 128,
                "stream": False,
            },
            timeout=120,
        )
        return (request_id, response.status_code, "")
    except Exception as exc:
        return (request_id, -1, str(exc))


def run_sequential(client, count: int) -> None:
    """Run count sequential requests."""
    log.info("=== Sequential stress test: %d requests ===", count)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a Python function to reverse a string.",
        "What are the primary colors?",
        "Describe the water cycle.",
    ]
    failures = 0
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        rid = f"seq-{i:04d}-{uuid.uuid4().hex[:8]}"
        start = time.perf_counter()
        rid, status, error = make_request(client, prompt, rid)
        elapsed = time.perf_counter() - start
        if status == 200:
            log.info("  [%s] OK (%d, %.1fs)", rid, status, elapsed)
        else:
            failures += 1
            log.error("  [%s] FAIL (%d, %.1fs): %s", rid, status, elapsed, error)
    log.info("Sequential: %d/%d passed, %d failed", count - failures, count, failures)


def run_concurrent(client, count: int, workers: int = 4) -> None:
    """Run count requests across workers threads."""
    log.info("=== Concurrent stress test: %d requests, %d workers ===", count, workers)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a Python function to reverse a string.",
        "What are the primary colors?",
        "Describe the water cycle.",
        "What is machine learning?",
        "Explain the theory of relativity briefly.",
        "Write a haiku about autumn.",
    ]
    results = []
    results_lock = threading.Lock()

    def worker(worker_id: int, num_requests: int) -> None:
        for i in range(num_requests):
            prompt = prompts[(worker_id * num_requests + i) % len(prompts)]
            rid = f"con-{worker_id}-{i:04d}-{uuid.uuid4().hex[:8]}"
            start = time.perf_counter()
            rid, status, error = make_request(client, prompt, rid)
            elapsed = time.perf_counter() - start
            with results_lock:
                results.append((rid, status, elapsed, error))
            if status == 200:
                log.info("  [%s] OK (%d, %.1fs)", rid, status, elapsed)
            else:
                log.error("  [%s] FAIL (%d, %.1fs): %s", rid, status, elapsed, error)

    threads = []
    per_worker = count // workers
    for w in range(workers):
        t = threading.Thread(target=worker, args=(w, per_worker), name=f"worker-{w}")
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    failures = sum(1 for r in results if r[1] != 200)
    log.info("Concurrent: %d/%d passed, %d failed", len(results) - failures, len(results), failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test for Cloud Server inference")
    parser.add_argument("--concurrent", action="store_true", help="Run concurrent test")
    parser.add_argument("--count", type=int, default=20, help="Number of requests")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from fastapi.testclient import TestClient
    from app.main import create_app

    log.info("Creating test app...")
    app = create_app()
    client = TestClient(app)

    if args.concurrent:
        run_concurrent(client, args.count, args.workers)
    else:
        run_sequential(client, args.count)

    log.info("Stress test complete.")


if __name__ == "__main__":
    main()
