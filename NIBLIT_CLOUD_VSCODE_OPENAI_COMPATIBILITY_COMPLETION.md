# NIBLIT CLOUD — VS CODE / OPENAI COMPATIBILITY COMPLETION REPORT

## 1. Source of remaining /v1/v1/models

**Caller: Continue extension** (`C:\Users\Riyaad\.continue\config.yaml`).
Two openai-provider entries had `apiBase: http://127.0.0.1:8000/v1`; Continue
composes `{apiBase}/v1/models`, producing `/v1/v1/models` on its ~30s model
poll. Fixed by setting `apiBase: http://127.0.0.1:8000` (server root) and
correcting model IDs (`llama3`, `qwen-coder2.5-3b` → advertised
`qwen2.5-coder-3b`). No `/v1/v1` compat route was added; live log confirms
zero doubled-prefix requests after the fix.

## 2. Source of CONNECT vscode.dev:443

**Caller: GitHub Copilot LLM Gateway extension.**
`github.copilot.llm-gateway.serverUrl` was set to
`http://127.0.0.1:8000/v1` in BOTH workspace settings files. That extension
treats `serverUrl` as a PROXY and tunnels all its traffic through it —
including HTTPS CONNECT tunnels to vscode.dev. No HTTP(S)_PROXY environment
variables existed; Cloud was never a configured system proxy.

Fix: removed the `serverUrl` override from
`GitHub/.vscode/settings.json` and `Niblit/.vscode/settings.json` (with
explanatory comments). Cloud is configured per-extension as an OpenAI
API base, never as a gateway proxy.

## 3. /v1/completions implementation

Added `POST /v1/completions` (+ `/completions`) in `app/main.py`. It converts
the prompt to a single user message and routes through the SAME
`_preflight_chat → handle_chat` path as chat completions — no second inference
engine. Supports `model`, `prompt`, `max_tokens`, `temperature`, `stream`.
Non-streaming returns OpenAI `text_completion` shape with REAL engine usage;
streaming emits `text_completion.chunk` SSE frames + `[DONE]`.

## 4. Streaming preflight repair

Root cause: `stream_chat_completion` invoked `handle_chat` lazily INSIDE the
generator, so governance/attention/context rejections raised
`HTTPException` after `StreamingResponse` had begun →
"response already started" RuntimeError.

Fix: extracted deterministic `_preflight_chat()` (context estimate, attention,
governance, temporal record, model resolution). The route now runs it EAGERLY
before constructing `StreamingResponse` for both chat and completion streams;
the generator receives the preflight result. Governed rejection ⇒ normal JSON
error, always before any bytes are streamed.

## 5–7. Token budget semantics

- `requested = prompt_tokens + max_tokens` estimate fed to governance.
- The 8192 value is an intentional governance ceiling
  (`NIBLIT_CG_MAX_TOKENS`) on per-request **completion** tokens — separate
  from context budgeting.
- Policy corrected: oversized `max_tokens` is now **CLAMPED to 8192 and
  recorded** (`rationale.token_limit.action="clamped"`) instead of rejected;
  `ModelManager` additionally clamps to the true budget
  (`effective_max_tokens = min(requested, n_ctx·guard − prompt − reserve)`),
  so the effective limit is min(model ctx, runtime n_ctx, governance max,
  provider max). `estimated_total_tokens > min(16384, context_window)`
  remains a HARD block (`GUARD_RUNAWAY_TOKENS`) — model context can never be
  silently exceeded.

## 8–9. Model ownership & advertised routing

| Model ID | Source | Provider | Backend | Available | Loaded | Advertised | Default |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder-3b | CloudConfig/ModelManager | llama_cpp (in-process) | GGUF qwen2.5-coder-3b-instruct-q4_k_m | yes | yes on demand | yes (/v1/models) | yes |
| docker.io/ai/qwen3:8B-Q4_K_M | ModelRegistry (Ollama live discovery) | ollama | Ollama daemon | only if daemon reachable | no (not routed through Cloud) | diagnostic APIs only | no |
| docker.io/ai/llama3.2:3B-Q4_K_M | ModelRegistry (Ollama live discovery) | ollama | Ollama daemon | only if daemon reachable | no | diagnostic APIs only | no |
| qwen_server (provider) | ProviderRegistry | managed llama-server | qwen_server.sh | unavailable (missing model/bin) | no | no | no |

The OpenAI execution list comes solely from the executable ModelManager, so
every advertised ID is routable (test-enforced). Registry/Ollama entries remain
visible via `/v1/models/detail` diagnostics without polluting the executable
advertisement.

## 10–14. Live results

Live probe (fresh instance, port 8011, real GGUF):
- `/v1/completions` non-streaming → 200 `{"object":"text_completion",
  "text":"OK", usage:{prompt:31, completion:1, total:32}}`
- `/v1/completions` streaming → 200, valid chunk frames + [DONE]
- Streaming chat with `max_tokens=13432` → **200 clean SSE stream**
  (previously crashed); clamp applied server-side
- Full log scan: **no Traceback, no "response already started", no /v1/v1/,
  no CONNECT, no guard_token rejections**
- OpenAI-shaped client exercised models + both completion endpoints with base
  `http://127.0.0.1:{port}/v1` composition semantics

VS Code configuration now in force:
- Copilot gateway `serverUrl` overrides removed (both scopes)
- Continue: `apiBase http://127.0.0.1:8000`, model `qwen2.5-coder-3b`
- Cline unchanged (`cline.openAiBaseUrl: http://127.0.0.1:8000/v1` — correct)
- Copilot provider model id aligned to `qwen2.5-coder-3b`

Note: full end-to-end verification inside a restarted VS Code session requires
reloading VS Code so the removed gateway override takes effect; protocol-level
behavior was verified with OpenAI-composition clients against the live server.

## Tests

New: `tests/test_vscode_openai_compat.py` — 7 tests: completions non-streaming,
completions streaming, oversized streaming chat stays clean + clamped,
non-streaming clamp applied, advertised==executable models, unknown-model
structured error, `/v1/v1/models` stays 404.
Updated: governance unit test documents clamp policy.
Full suite: **262 passed** (was 255).

## Files modified

- `app/main.py`: `_preflight_chat` extraction; eager preflight for streams;
  `/v1/completions` endpoint; `CompletionRequest.max_tokens/stream`
- `app/cloud_governance.py`: token-limit clamp policy
- `tests/test_vscode_openai_compat.py` (new), `tests/test_cognitive_runtime.py`
- VS Code: workspace + Niblit `.vscode/settings.json`,
  user `settings.json`, `~/.continue/config.yaml`

## Remaining limitations

1. In-process GGUF generation buffers then emits one content chunk (true
   incremental token streaming is a llama-cpp integration task, out of scope).
2. Ollama-discovered models are not executable THROUGH Cloud until a routing
   bridge is added; they are excluded from the executable advertisement.
3. `qwen_server` managed backend remains unconfigured/unavailable by design
   (missing model/bin paths on this machine).

## Verdict

- OPENAI CHAT API: **PASS**
- OPENAI COMPLETIONS API: **PASS**
- STREAMING: **PASS**
- TOKEN GOVERNANCE: **CORRECT** (clamp + expose; hard block preserved for context overflow)
- STREAM ERROR HANDLING: **PASS**
- MODEL OWNERSHIP: **CONSISTENT**
- VS CODE PROXY CONFIG: **CORRECT** (gateway misuse removed)
- VS CODE → CLOUD: **PASS** (protocol-verified; final in-editor check after VS Code reload)

