# NIBLIT CLOUD — INFERENCE HANG ROOT-CAUSE REPAIR

## WORK COMPLETED BEFORE THIS PHASE

Prior sessions established Cloud as the authoritative inference layer, fixed
`/v1` URL composition, implemented `/v1/completions`, repaired streaming
preflight, converged the local GGUF registry (qwen2.5-coder-3b / llama3.2-1b /
qwen2.5-0.5b), and added a Cloud Web UI. Despite all of this, a direct
non-stream `POST /v1/chat/completions` probe timed out on isolated instances
even though native `llama_cpp.Llama` provably loaded qwen2.5-coder-3b and
generated text. The first blocking boundary had not been identified.

## ROOT CAUSE

**Self-deadlock on a non-reentrant `threading.Lock` inside `ModelManager`.**

- `ModelManager.chat()` (`app/main.py`) entered `with self._lock:`
- Inside that critical section it constructed `GGUFEngine(...)` passing
  `n_ctx=self.model_context_limit(model_id)`
- `model_context_limit()` itself does `with self._lock:` — re-acquiring the
  **same non-reentrant** lock → permanent block.

This regressed when per-model context limits were introduced in the multi-GGUF
phase. Only **cold-engine** requests deadlock (engine not yet loaded); warm
models never reach that code path, which is why health/models endpoints stayed
fine, native llama.cpp stayed healthy, and clean instances reproduced the hang.

## FIRST BLOCKING FUNCTION

`ModelManager.model_context_limit()` at `app/main.py:370`, called from
`chat()` line 466 while holding the manager lock.

## HOW IT WAS PROVEN

1. Boundary instrumentation (START/END logs) already present from the
   interrupted session showed the request stopping immediately after
   `preflight[...] governance done` with no `Loading model from ...` log.
2. Server process was idle (~0 CPU, small RAM) — blocked, not computing.
3. `py-spy dump --pid <server>` captured the definitive stack:

```
model_context_limit (app\main.py:370)   ← blocked acquiring self._lock
chat (app\main.py:466)
handle_chat (app\main.py:1031)
chat_completions (app\main.py:1148)
```

## BEFORE CALL GRAPH

request → preflight (ok) → handle_chat → manager.chat()
  → `with self._lock:` → GGUFEngine(n_ctx=self.model_context_limit(id))
  → model_context_limit() → `with self._lock:` → **DEADLOCK**

## AFTER CALL GRAPH

request → preflight (ok) → handle_chat → manager.chat()
  → ctx_limit = model_context_limit(id)   (lock taken + released)
  → `with self._lock:` → GGUFEngine(n_ctx=ctx_limit) → engine.chat()
  → llama_cpp generation → OpenAI response

## FILES MODIFIED

- `app/main.py` — moved `model_context_limit()` call outside the manager-lock
  critical section in `chat()`; demoted temporary per-stage preflight logs to
  DEBUG (cognitive-gateway complete-request summary remains the INFO diagnostic).

## FILES ADDED

- `tests/test_inference_hang_regression.py` — two tests:
  cold-engine chat must complete within 15 s instead of deadlocking, engine is
  created with the model-specific context limit; warm path still works with no
  duplicate engine loads. Fails against the previous behavior (thread never
  joins), passes after the repair.

## TEST RESULTS

- `pytest tests -q` → **282 passed** (incl. the new regression tests).
- No existing test was weakened; no governance check removed.

## LIVE VALIDATION (isolated instance, port 8017, project venv)

- Boot → `/healthz` 200, `/health` 200, Inference READY.
- `POST /v1/chat/completions`, model=qwen2.5-coder-3b, stream=false,
  max_tokens=16 → **200 in 11.4 s**, content=`'OK'`,
  usage={prompt_tokens:33, completion_tokens:1}. Log shows full chain:
  cognitive classify → explicit model selection → preflight → ModelManager
  load (`qwen2.5-coder-3b-instruct-q4_k_m.gguf`) → Inference OK.
- llama3.2-1b non-stream → **200 in 16.0 s**, content=`'OK'`.
- qwen2.5-0.5b non-stream → **200 in 2.3 s**, content=`'OK'`.
- Streaming (default model) → **200**, valid SSE frames + `[DONE]`,
  assembled text `"Hello! It's a pleasure to meet you!"`.
- No ASGI tracebacks, no "response already started", no `/v1/v1/` regression
  caused by this work.

## VERDICTS

- qwen2.5-coder-3b non-stream: **PASS**
- qwen2.5-coder-3b stream: **PASS**
- llama3.2-1b: **PASS**
- qwen2.5-0.5b: **PASS**
- Cloud Web UI: **NOT TESTED this phase** (direct API ladder takes precedence; UI uses the same endpoints now proven working)
- Core → Cloud: **NOT TESTED** (requires running Core; next step in the ladder)
- Niblit UI → Core → Cloud: **NOT TESTED**
- /v1/v1: **DEFERRED** (known separate client-side issue — Continue apiBase; previously fixed config remains)
- CONNECT vscode.dev: **DEFERRED** (Copilot LLM gateway proxy override previously removed from settings; VS Code reload pending)
- TradingRuntimeManager authority: **PASS** (untouched; Cloud's lean_algo_manager remains a compatibility/health probe only)
- LeanAlgoManager compatibility boundary: **PASS** (unchanged)
- Inference → trading isolation: **PASS** (no inference-to-execution path introduced)

## REMAINING LADDER STEPS

Cloud Web UI smoke → Niblit Core round-trip → niblit-ui end-to-end → confirm
no residual `/v1/v1` or CONNECT callers after VS Code reload.
