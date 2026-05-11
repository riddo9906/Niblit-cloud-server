# Niblit Cognitive Cloud Runtime (Phase Ω.7)

A **governed, observable, adaptive multi-model inference runtime** — the distributed cognitive execution layer for the Niblit ecosystem.

> **Backward compatible**: all existing HuggingFace-style, llama.cpp, and QwenLocalBrain-compatible APIs are fully preserved.

## What changed in Phase Ω.7

The server is no longer just a GGUF inference endpoint.  It has evolved into:

| Layer | Module | Purpose |
|---|---|---|
| **Model Orchestration** | `app/model_orchestrator.py` | Trust-scored, latency-aware multi-model routing |
| **Cognitive Envelope** | `app/cognitive_envelope.py` | Schema-v2 request enrichment (backward compat) |
| **Constitutional Governance** | `app/cloud_governance.py` | Seven constitutional laws + cloud safety guards |
| **Temporal Coherence** | `app/temporal_sync.py` | Epoch tracking, coherence EMA, drift detection |
| **Reflection Engine** | `app/reflection_engine.py` | Quality telemetry, JSONL persistence, auto-reflection |
| **Attention Economy** | `app/attention_allocator.py` | Salience-weighted request prioritization |
| **Trading Bridge** | `app/trading_runtime_bridge.py` | niblit-lean-algos signal/regime integration |
| **Event Bus** | `app/event_bus.py` | Structured events aligned with Niblit Ω.7 constants |
| **Node Identity** | `app/node_identity.py` | Stable fingerprint, cluster/swarm readiness stubs |

## Features

- Hugging Face-style chat completions endpoint (`/v1/chat/completions`)
- Hugging Face-style inference endpoint (`/models/{model}`)
- **Drop-in llama-server replacement** for Niblit's `QwenLocalBrain` HTTP backend
  - Handles `"model": "local"` alias automatically
  - Responds to all probe endpoints: `GET /health`, `GET /v1/models`, `GET /props`
  - Supports legacy `POST /completion` endpoint
- Multi-model orchestration with trust + latency scoring
- Optional cognitive envelope (schema v2) — plain requests still work
- Constitutional governance (7 laws + cloud safety guards)
- Temporal epoch synchronization
- Cognitive telemetry via JSONL reflection logs
- Attention economy (salience-based request prioritization)
- Trading cognition bridge (reads niblit-lean-algos signal files)
- Structured event bus aligned with Niblit Phase Ω.7
- Node identity + cluster readiness stubs for future swarm cognition

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Connecting Niblit to this server

Set the following environment variables in your **Niblit** deployment:

```bash
NIBLIT_LLAMA_SERVER_URL=https://<your-cloud-server>.fly.dev
NIBLIT_GGUF_BACKEND=http
NIBLIT_LLAMA_SERVER_TIMEOUT=300
```

Niblit's `QwenLocalBrain` will then:
1. Probe `https://<your-cloud-server>/health` → `200 OK` ✅
2. Call `POST /v1/chat/completions` with `"model": "local"` → handled ✅
3. Fall back to `POST /completion` if needed → handled ✅

## Configuration

See `.env.example` for all options.

### Core

| Variable | Default | Description |
|---|---|---|
| `GGUF_MODELS_JSON` | `{}` | JSON map of `model_id → gguf_path` |
| `DEFAULT_MODEL_ID` | first model | Default model if request omits model |
| `COMPAT_PREFIXES` | `hf,local,kimi,claude` | Compatibility URL prefixes |
| `N_CTX` | `4096` | llama.cpp context length |
| `N_THREADS` | `4` | llama.cpp thread count |

### Cognitive Runtime (Phase Ω.7)

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_MO_ENABLED` | `1` | Enable model orchestrator |
| `NIBLIT_CG_ENABLED` | `1` | Enable constitutional governance |
| `NIBLIT_CG_STRICT` | `1` | Strict mode (blocks violations) |
| `NIBLIT_CG_MAX_TOKENS` | `8192` | Hard token ceiling |
| `NIBLIT_RE_ENABLED` | `1` | Enable reflection engine |
| `NIBLIT_AA_ENABLED` | `1` | Enable attention economy |
| `NIBLIT_TRADING_BRIDGE_ENABLED` | `1` | Enable trading bridge |

## API

### Existing endpoints (unchanged)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health`, `/healthz` | Health probe |
| `GET` | `/props` | Legacy llama-server probe |
| `GET` | `/v1/models` | List models |
| `GET` | `/v1/models/{id}` | Get model info |
| `POST` | `/v1/chat/completions` | Chat completions (OpenAI-style) |
| `POST` | `/completion` | Legacy llama-server completion |
| `POST` | `/models/{model}` | HuggingFace inference API |
| `POST` | `/{prefix}/...` | Compatibility prefix routes |

### New cognitive endpoints (Phase Ω.7)

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/cognitive/chat` | Enriched chat with cognitive metadata |
| `GET` | `/v1/runtime/status` | Full runtime snapshot |
| `GET` | `/v1/runtime/coherence` | Temporal coherence state |
| `GET` | `/v1/runtime/governance` | Constitutional governance stats |
| `GET` | `/v1/runtime/attention` | Attention economy metrics |
| `GET` | `/v1/runtime/models` | Model orchestration health |
| `GET` | `/v1/runtime/reflection` | Reflection engine telemetry |
| `GET` | `/v1/runtime/trading` | Trading cognition bridge state |
| `GET` | `/v1/runtime/epoch` | Current epoch and coherence |
| `GET` | `/metrics/cognitive` | Cognitive telemetry metrics |
| `GET` | `/metrics/coherence` | Coherence metrics |
| `GET` | `/metrics/governance` | Governance metrics |
| `GET` | `/metrics/models` | Model health metrics |
| `GET` | `/cluster/status` | Cluster status (single-node) |
| `GET` | `/cluster/identity` | Node identity |
| `GET` | `/cluster/capabilities` | Node capabilities |

### Cognitive Envelope (optional)

Any chat request can optionally include cognitive envelope fields.  Plain
requests that omit these fields are fully backward compatible:

```json
{
  "model": "qwen2.5-0.5b",
  "messages": [{"role": "user", "content": "Hello"}],
  "intent": "analytical",
  "execution_mode": "balanced",
  "coherence_score": 0.91,
  "constitutional_priority": "safety",
  "attention_budget": 0.8,
  "resource_mode": "balanced",
  "epoch_tag": "epoch_4421",
  "governance": {"governance_mode": "normal"},
  "temporal": {"coherence_score": 0.91}
}
```

## Architecture

See [architecture.md](architecture.md) for a full system diagram and layer descriptions.

## Deployment

See [deployment.md](deployment.md) for Docker, Fly.io, and HuggingFace Spaces instructions.

## Compatibility notes

- All existing provider routes (`hf`, `local`, `kimi`, `claude`) continue to work.
- The `model: "local"` alias from Niblit's `QwenLocalBrain` is still handled.
- Constitutional governance is **strict by default** — set `NIBLIT_CG_STRICT=0` for permissive (log-only) mode during migration.

