#!/usr/bin/env python3
"""Standalone llama-cpp-python inference validator.

Loads the exact same GGUF model used by the Cloud Server, tokenizes a short
prompt, generates a few tokens, and prints every stage.  No FastAPI, no
CloudRuntime, no provider abstraction — pure library test.

Usage:
    python -m app.debug.llama_validation

If this script segfaults: the problem is in llama-cpp-python or the GGUF file.
If it succeeds: the Cloud Server wrapper is passing something wrong.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("llama_validation")


def main() -> None:
    # ── Load the SAME configuration the Cloud Server uses ────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.config import get_config

    cfg = get_config()
    models = cfg.model_map
    if not models:
        log.error("No models found in configuration")
        sys.exit(1)

    model_id, model_path = next(iter(models.items()))
    log.info("Configuration loaded")
    log.info("  model_id:   %s", model_id)
    log.info("  model_path: %s", model_path)
    log.info("  n_ctx:      %d", cfg.n_ctx)
    log.info("  n_threads:  %d", cfg.n_threads)

    if not os.path.isfile(model_path):
        log.error("Model file not found: %s", model_path)
        sys.exit(1)

    # ── Stage 1: Safest possible Llama() constructor ────────────────────
    log.info("Importing llama_cpp...")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        log.error("llama-cpp-python not installed: %s", exc)
        sys.exit(1)

    import llama_cpp
    log.info("llama_cpp version: %s", getattr(llama_cpp, "__version__", "unknown"))

    kwargs = {
        "model_path": model_path,
        "n_ctx": 4096,           # safest: small context
        "n_threads": 1,          # safest: single thread
        "n_gpu_layers": 0,       # safest: CPU only
        "use_mmap": False,       # Windows safety
        "use_mlock": False,      # safest: no mlock
        "seed": 42,
        "verbose": False,
        "offload_kqv": False,     # safest: no offloading
        "flash_attn": False,      # safest: no flash attention
    }
    log.info("Stage 1: Constructing Llama() with kwargs:")
    for k, v in kwargs.items():
        log.info("  %s = %s", k, v)

    try:
        llm = Llama(**kwargs)
    except Exception as exc:
        log.error("STAGE 1 FAILED: Llama() constructor raised: %s", exc)
        sys.exit(1)

    log.info("Stage 1 PASSED: Model loaded successfully")

    # ── Stage 2: Tokenizer round-trip ───────────────────────────────────
    log.info("Stage 2: Tokenizer round-trip")
    test_text = "Hello"
    try:
        tokens = llm.tokenize(test_text.encode("utf-8"))
        decoded = llm.detokenize(tokens).decode("utf-8", errors="replace")
        log.info("  tokenize('%s') -> %s", test_text, tokens)
        log.info("  detokenize(%s) -> '%s'", tokens[:10], decoded)
        assert decoded.strip() == test_text, f"Round-trip failed: '{decoded}' != '{test_text}'"
    except Exception as exc:
        log.error("STAGE 2 FAILED: %s", exc)
        sys.exit(1)
    log.info("Stage 2 PASSED: Tokenizer round-trip OK")

    # ── Stage 3: Basic tokenization of a prompt ─────────────────────────
    log.info("Stage 3: Tokenize prompt")
    prompt = "What is the capital of France?"
    try:
        prompt_bytes = prompt.encode("utf-8")
        prompt_tokens = llm.tokenize(prompt_bytes)
        log.info("  prompt: '%s'", prompt)
        log.info("  prompt length (chars): %d", len(prompt))
        log.info("  prompt tokens: %d", len(prompt_tokens))
        log.info("  first 20 token ids: %s", prompt_tokens[:20])
        log.info("  last 20 token ids:  %s", prompt_tokens[-20:])
    except Exception as exc:
        log.error("STAGE 3 FAILED: %s", exc)
        sys.exit(1)
    log.info("Stage 3 PASSED: Tokenization OK")

    # ── Stage 4: Raw completion (n_predict=8) ───────────────────────────
    log.info("Stage 4: Raw completion (create_completion)")
    log.info("  n_predict=8, temperature=0.0, top_k=1 (greedy)")
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=8,
            temperature=0.0,
            top_k=1,
            stop=[],
            stream=False,
        )
        text = result.get("choices", [{}])[0].get("text", "")
        log.info("  completion text: '%s'", text)
        log.info("  full result keys: %s", list(result.keys()))
    except Exception as exc:
        log.error("STAGE 4 FAILED: create_completion raised: %s", exc)
        sys.exit(1)
    log.info("Stage 4 PASSED: Raw completion succeeded")

    # ── Stage 5: Chat completion (n_predict=8) ──────────────────────────
    log.info("Stage 5: Chat completion (create_chat_completion)")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        chat_result = llm.create_chat_completion(
            messages=messages,
            max_tokens=8,
            temperature=0.0,
            top_k=1,
            stop=[],
            stream=False,
        )
        chat_text = chat_result.get("choices", [{}])[0].get("message", {}).get("content", "")
        log.info("  chat completion text: '%s'", chat_text)
    except Exception as exc:
        log.error("STAGE 5 FAILED: create_chat_completion raised: %s", exc)
        sys.exit(1)
    log.info("Stage 5 PASSED: Chat completion succeeded")

    # ── Stage 6: Chat with longer prompt ────────────────────────────────
    log.info("Stage 6: Chat completion with longer prompt")
    long_prompt = (
        "Write a detailed explanation of how neural networks work, "
        "including forward propagation, backpropagation, and gradient descent."
    )
    long_messages = [
        {"role": "system", "content": "You are an AI expert."},
        {"role": "user", "content": long_prompt},
    ]
    try:
        long_tokens_est = len(long_prompt) // 4
        log.info("  prompt tokens (est): %d", long_tokens_est)
        long_result = llm.create_chat_completion(
            messages=long_messages,
            max_tokens=16,
            temperature=0.0,
            top_k=1,
            stop=[],
            stream=False,
        )
        long_text = long_result.get("choices", [{}])[0].get("message", {}).get("content", "")
        log.info("  chat completion text (first 100 chars): '%s'", long_text[:100])
    except Exception as exc:
        log.error("STAGE 6 FAILED: %s", exc)
        sys.exit(1)
    log.info("Stage 6 PASSED: Longer chat completion succeeded")

    # ── All stages passed ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("ALL STAGES PASSED")
    log.info("llama-cpp-python %s is working correctly with model: %s",
             getattr(llama_cpp, "__version__", "?"), model_id)
    log.info("=" * 60)

    # Return inference stats
    print(f"\nRESULT: VALIDATION_SUCCESS")
    print(f"Model: {model_id}")
    print(f"Path: {model_path}")
    print(f"Version: {getattr(llama_cpp, '__version__', 'unknown')}")
    print(f"Prompt tokens (stage 3): {len(prompt_tokens)}")
    print(f"Completion (stage 4): '{text}'")
    print(f"Chat (stage 5): '{chat_text}'")


if __name__ == "__main__":
    main()