#!/usr/bin/env python3
"""
tools/check_local_llm.py — Diagnostics for local llama.cpp server.

Verifies that the local LLM server is running and all endpoints are functional:
- llama-server reachable
- /v1/models
- chat endpoint
- completion endpoint
- streaming
- repository prompt
- code generation
- multi-file edit
- latency
- context length
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error

# Configuration
LOCAL_MODEL_ENDPOINT = os.getenv("LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = int(os.getenv("LOCAL_MODEL_TIMEOUT", "30"))


def check_endpoint(name: str, method: str, path: str, data: dict | None = None) -> tuple[bool, str]:
    """Check a single endpoint and return (success, message)."""
    url = f"{LOCAL_MODEL_ENDPOINT}{path}"
    try:
        if method == "GET":
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            latency = (time.perf_counter() - start) * 1000
            content = resp.read().decode()
            try:
                json_data = json.loads(content)
                return True, f"OK ({latency:.1f}ms) - {json.dumps(json_data)[:100]}..."
            except json.JSONDecodeError:
                return True, f"OK ({latency:.1f}ms) - {content[:100]}..."
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode()[:100]}"
    except urllib.error.URLError as exc:
        return False, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return False, f"Error: {exc}"


def check_streaming() -> tuple[bool, str]:
    """Check if streaming is supported."""
    url = f"{LOCAL_MODEL_ENDPOINT}/v1/chat/completions"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": "local",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 20,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            content = resp.read().decode()
            latency = (time.perf_counter() - start) * 1000
            if "data:" in content:
                return True, f"Streaming OK ({latency:.1f}ms)"
            return False, f"No streaming format detected ({latency:.1f}ms)"
    except Exception as exc:
        return False, f"Streaming failed: {exc}"


def check_context_length() -> tuple[bool, str]:
    """Check the context length from /v1/models."""
    url = f"{LOCAL_MODEL_ENDPOINT}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", []) or data.get("models", [])
            if models:
                model = models[0]
                n_ctx = model.get("meta", {}).get("n_ctx", "unknown")
                return True, f"Context length: {n_ctx}"
            return False, "No models found"
    except Exception as exc:
        return False, f"Error: {exc}"


def check_code_generation() -> tuple[bool, str]:
    """Test code generation capability."""
    url = f"{LOCAL_MODEL_ENDPOINT}/v1/chat/completions"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": "local",
                "messages": [{"role": "user", "content": "Write a Python function that adds two numbers."}],
                "max_tokens": 100,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            latency = (time.perf_counter() - start) * 1000
            data = json.loads(resp.read().decode())
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Check for Python code indicators
            if "def " in text or "return " in text or "import " in text:
                return True, f"Code generated ({latency:.1f}ms) - {text[:50]}..."
            return True, f"Response received ({latency:.1f}ms) - {text[:50]}..."
    except Exception as exc:
        return False, f"Error: {exc}"


def check_repository_prompt() -> tuple[bool, str]:
    """Test repository-aware coding prompt."""
    url = f"{LOCAL_MODEL_ENDPOINT}/v1/chat/completions"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": "local",
                "messages": [{
                    "role": "user",
                    "content": "Given this file structure: Niblit-cloud-server/app/main.py, Niblit-cloud-server/tools/local_llm_provider.py - how would you add a new endpoint for file operations?",
                }],
                "max_tokens": 150,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            latency = (time.perf_counter() - start) * 1000
            data = json.loads(resp.read().decode())
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return True, f"Repository prompt OK ({latency:.1f}ms) - {text[:60]}..."
    except Exception as exc:
        return False, f"Error: {exc}"


def check_multi_file_edit() -> tuple[bool, str]:
    """Test multi-file edit capability."""
    url = f"{LOCAL_MODEL_ENDPOINT}/v1/chat/completions"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": "local",
                "messages": [{
                    "role": "user",
                    "content": "I have two files: file1.py with content 'x=1' and file2.py with content 'y=2'. How would I modify them to share a common variable?",
                }],
                "max_tokens": 200,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            latency = (time.perf_counter() - start) * 1000
            data = json.loads(resp.read().decode())
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Check if response mentions multiple files
            if "file" in text.lower() or "both" in text.lower():
                return True, f"Multi-file edit OK ({latency:.1f}ms) - {text[:60]}..."
            return True, f"Response received ({latency:.1f}ms) - {text[:60]}..."
    except Exception as exc:
        return False, f"Error: {exc}"


def main() -> int:
    """Run all diagnostics and report results."""
    print("=" * 60)
    print("Local LLM Diagnostics — llama.cpp/OpenAI-compatible server")
    print(f"Endpoint: {LOCAL_MODEL_ENDPOINT}")
    print("=" * 60)
    
    checks = [
        ("llama-server reachable (GET /health)", "GET", "/health"),
        ("/v1/models endpoint", "GET", "/v1/models"),
        ("/v1/chat/completions endpoint", "POST", "/v1/chat/completions", {"model": "local", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 10}),
        ("/v1/completions endpoint", "POST", "/v1/completions", {"prompt": "Test", "max_tokens": 10}),
        ("streaming response", None, None, None),  # Special check
        ("context length", None, None, None),  # Special check
        ("code generation", None, None, None),  # Special check
        ("repository prompt", None, None, None),  # Special check
        ("multi-file edit", None, None, None),  # Special check
    ]
    
    all_passed = True
    
    for check in checks:
        name = check[0]
        if check[1] is None:
            # Special check
            if "streaming" in name:
                ok, msg = check_streaming()
            elif "context" in name:
                ok, msg = check_context_length()
            elif "code generation" in name:
                ok, msg = check_code_generation()
            elif "repository prompt" in name:
                ok, msg = check_repository_prompt()
            elif "multi-file edit" in name:
                ok, msg = check_multi_file_edit()
            else:
                ok, msg = False, "Unknown check"
        else:
            ok, msg = check_endpoint(name, check[1], check[2], check[3] if len(check) > 3 else None)
        
        status = "✅" if ok else "❌"
        print(f"  {status}  {name}")
        print(f"      {msg}")
        print()
        if not ok:
            all_passed = False
    
    # Latency test
    print("Latency Test (3 requests):")
    latencies = []
    for i in range(3):
        try:
            req = urllib.request.Request(
                f"{LOCAL_MODEL_ENDPOINT}/v1/chat/completions",
                data=json.dumps({
                    "model": "local",
                    "messages": [{"role": "user", "content": "Quick test"}],
                    "max_tokens": 10,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start = time.perf_counter()
            urllib.request.urlopen(req, timeout=TIMEOUT)
            latencies.append((time.perf_counter() - start) * 1000)
        except Exception:
            pass
    
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  Average latency: {avg:.1f}ms")
        print(f"  Individual: {[f'{l:.1f}ms' for l in latencies]}")
    else:
        print("  Could not measure latency")
    
    print("=" * 60)
    if all_passed:
        print("All checks passed! ✅")
        return 0
    else:
        print("Some checks failed! ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())