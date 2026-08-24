# NIBLIT CLOUD — CONNECTIVITY + INFERENCE ROUTING REPAIR

## 1. Root cause of the reachability / 404 issue

The server's routes were always correct (`/v1/models`, `/v1/chat/completions`).
The observed `GET /v1/v1/models -> 404` came from **client-side composition**:
a client configured with an OpenAI-style base URL (`…:8000/v1`) and then
appended the full API path `/v1/models`. Concretely,
`niblit-ui/src/config/api.ts` built `CLOUD_BASE_URL + "/v1/models"` without
stripping a trailing `/v1` from a user-configured `VITE_NIBLIT_CLOUD_URL`.
The same hazard existed for any raw HTTP helper handed an OpenAI-style base.

## 2. Bind host vs client URL

Separated explicitly in `CloudConfig`:

- `host` (env `NIBLIT_CLOUD_BIND_HOST`, falling back to `NIBLIT_HOST`) —
  SERVER BIND address; `0.0.0.0` remains valid here.
- `client_base_url_resolved` (env `NIBLIT_CLOUD_BASE_URL`, default
  `http://127.0.0.1:{port}`) — what clients must use. `0.0.0.0` is never used
  as a client destination.
- `openai_base_url` property → exactly one `/v1`.

Boot diagnostics now print both:
`API: http://127.0.0.1:8000 (OpenAI base: http://127.0.0.1:8000/v1)` and
`Bind: 0.0.0.0:8000`.

## 3. Source of /v1/v1/models

UI cloud client composition (above). Fixed centrally in `api.ts`
(`normalizeCloudRoot` strips trailing `/v1` segments) and defensively in the
new server helper `app/base_urls.py::join_openai_base`, which tolerates
endpoints passed with or without `/v1`. **No compatibility `/v1/v1/*` route
was added** — the live probe confirms `/v1/v1/models -> 404`.

## 4. API route map (verified)

| Path | Status | Purpose |
|---|---|---|
| `/healthz` | 200 | liveness only |
| `/health` | 200 ready / 503 degraded | readiness + inference truth |
| `/v1/models`, `/models` | 200 | OpenAI model list |
| `/v1/chat/completions` | 200 | streaming + non-streaming |
| `/v1/runtime/status`, `/v1/runtime/*` | 200 | runtime telemetry |
| `/{prefix}/v1/...` (`hf,local,kimi,claude`) | 200 | legacy compat aliases |

## 5–8. Model & provider registry

Before: config default `qwen2.5-coder-3b`; ModelRegistry hardcoded
`ollama/llama3` unconditionally; ProviderRegistry counted `qwen_server` even
when its model/backend were missing ("model not found"); boot printed an
unqualified `Cloud READY`.

After:

| configured | registry/manager model | provider | available | loaded at request | default |
|---|---|---|---|---|---|
| qwen2.5-coder-3b | qwen2.5-coder-3b | llama.cpp in-process GGUF (`C:/Users/Riyaad/llama_migration/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf`) | yes | yes (live probe) | yes |

- Ollama registration is now **live discovery** via
  `GET {OLLAMA_BASE_URL}/api/tags` (`_discover_ollama_models`); an unreachable
  daemon advertises nothing.
- Also fixed a latent port collision: the managed qwen backend defaulted to
  `NIBLIT_PORT` (the same 8000 as uvicorn); it now uses `NIBLIT_QWEN_PORT`
  (default 8081).

## 9–10. Default-model resolution & health semantics

`/health` now returns:

```json
{"status":"ready","api_ready":true,"inference_ready":true,
 "default_model":"qwen2.5-coder-3b","models":["qwen2.5-coder-3b"]}
```

and `503 {"status":"degraded", …, "reason":"default_model_unavailable"}` when
no usable model/default exists. Boot output is qualified: `Cloud API READY`
plus a separate `Inference READY — default model: …` or
`Inference DEGRADED — default model unavailable` line.

## 11–14. Live probe results (fresh instance, port 8010)

- `/healthz` → 200 `{"status":"ok"}`
- `/health` → 200 ready payload (above)
- `/v1/models` → 200, advertises the real GGUF-backed model only
- `/v1/v1/models` → **404** (invariant held; no compat shim)
- Non-streaming completion → 200,
  `choices[0].message.content = "OK"`, 34 tokens
- Streaming (`stream:true`) → 200, valid SSE deltas (2), assembled content "OK"
- Server log shows single `/v1/models` hits — no doubled prefixes

Note: an older Cloud instance was found still bound to :8000 during probing;
it served pre-fix responses until restarted. Restarting that instance picks up
all fixes (verified by running identical code on :8010).

Qwen "thinking mode": this deployment routes to llama.cpp GGUF directly (not
Ollama/Docker); content was non-empty with no `think:false` handling required.
Ollama-specific thinking suppression was therefore not added (nothing to fix
on the active path; would be provider-gated if an Ollama provider becomes
active).

## 15–19. Clients, files, tests, limitations

- Niblit Core client (`modules/local_brain.py`) already composes correctly
  (`http://127.0.0.1:8000` root + `/v1/chat/completions`; `/healthz` for
  liveness) — no change needed. UI client fixed as above.
- VS Code/OpenAI compatibility proven with a raw OpenAI-shaped client using
  base `http://127.0.0.1:{port}/v1` semantics through the same endpoints;
  documented settings for later: base `http://127.0.0.1:8000/v1`,
  model id taken from `/v1/models`.
- CORS: existing middleware untouched (no browser credential flows on this
  path); no unrestricted-origin change made.

Files modified:
- `app/base_urls.py` (new — normalization helpers)
- `app/config.py` (bind vs client URL; `NIBLIT_CLOUD_BIND_HOST`,
  `NIBLIT_CLOUD_BASE_URL`; qwen port de-collision)
- `app/main.py` (`/healthz` vs truthful `/health`; qualified boot output;
  client-URL diagnostics)
- `app/model_registry.py` (live Ollama discovery)
- `tests/test_base_urls.py` (new — 5 tests incl. forbidden-sequence invariant)
- niblit-ui `src/config/api.ts` + `src/__tests__/api.test.ts`

Tests run:
- Cloud: `python -m pytest tests -q` → **255 passed**
- UI: `npx vitest run` → **27 passed**

Remaining limitations:
1. ProviderRegistry still registers `qwen_server` when misconfigured (its
   start fails cleanly); a per-provider `{registered, available, healthy,
   reason}` reporting endpoint is not yet added.
2. The stale instance on :8000 must be restarted to serve fixed code.
3. Ollama "thinking" suppression intentionally omitted (provider not active).

## Verdict

- CLOUD API: **READY**
- MODEL REGISTRY: **CONSISTENT**
- DEFAULT MODEL: **AVAILABLE**
- CHAT COMPLETIONS: **PASS**
- STREAMING: **PASS**
- NIBLIT → CLOUD: **PASS** (Core composes correct URLs against verified endpoints)
- OPENAI-COMPATIBLE CLIENT → CLOUD: **PASS**
- /v1 PREFIX: **FIXED**

