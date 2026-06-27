# Niblit Cognitive Cloud Runtime — Architecture (Phase Ω.9)

## Overview

The Niblit Cognitive Cloud Runtime transforms a simple GGUF inference endpoint
into a **governed, observable, topology-aware runtime orchestration node**.

Authority boundaries:
- `riddo9906/Niblit`: governance/cognition authority
- `riddo9906/Niblit-cloud-server`: runtime orchestration authority
- `riddo9906/niblit-lean-algos`: execution cognition authority

```
┌─────────────────────────────────────────────────────────────────┐
│                  Niblit Cognitive Cloud Runtime                  │
│                         Phase Ω.7                               │
├─────────────────────────────────────────────────────────────────┤
│  Inbound API Layer (FastAPI)                                    │
│  ┌──────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │ /v1/chat/        │ │ /v1/cognitive/  │ │ /v1/runtime/    │  │
│  │ completions      │ │ chat            │ │ *               │  │
│  │ /completion      │ │                 │ │ /metrics/*      │  │
│  │ /models/{model}  │ │                 │ │ /cluster/*      │  │
│  └─────────┬────────┘ └────────┬────────┘ └────────┬────────┘  │
│            └──────────────────┬┘                   │           │
│                               ▼                    │           │
│  ┌──────────────────────────────────────────────┐  │           │
│  │           Cognitive Envelope Parser          │  │           │
│  │           app/cognitive_envelope.py          │  │           │
│  │  schema-v2 normalization + backward compat   │  │           │
│  └──────────────────┬───────────────────────────┘  │           │
│                     ▼                              │           │
│  ┌──────────────────────────────────────────────┐  │           │
│  │        Attention Allocator                   │  │           │
│  │        app/attention_allocator.py            │  │           │
│  │  salience scoring → queue → overload guard   │  │           │
│  └──────────────────┬───────────────────────────┘  │           │
│                     ▼                              │           │
│  ┌──────────────────────────────────────────────┐  │           │
│  │      Constitutional Cloud Governance         │  │           │
│  │        app/cloud_governance.py               │  │           │
│  │  7 laws + token limit + injection detection  │  │           │
│  └──────────────────┬───────────────────────────┘  │           │
│                     ▼                              │           │
│  ┌──────────────────────────────────────────────┐  │           │
│  │         Model Orchestrator                   │  │           │
│  │        app/model_orchestrator.py             │  │           │
│  │  trust-scored + latency EMA routing          │  │           │
│  └──────────────────┬───────────────────────────┘  │           │
│                     ▼                              │           │
│  ┌──────────────────────────────────────────────┐  │           │
│  │         GGUF Inference Engine                │  │           │
│  │         GGUFEngine + ModelManager            │  │           │
│  │         (llama-cpp-python)                   │  │           │
│  └──────────────────┬───────────────────────────┘  │           │
│                     │                              │           │
│  ┌──────────────────▼───────────────────────────┐  │           │
│  │         Cross-Cutting Runtime Subsystems     ◄──┘           │
│  │  ┌────────────────┐  ┌─────────────────────┐ │             │
│  │  │ Temporal Sync  │  │  Reflection Engine  │ │             │
│  │  │ (epoch + EMA)  │  │  (JSONL + telemetry)│ │             │
│  │  └────────────────┘  └─────────────────────┘ │             │
│  │  ┌────────────────┐  ┌─────────────────────┐ │             │
│  │  │ Trading Bridge │  │   Event Bus (Ω.7)   │ │             │
│  │  │ (signal + JSONL│  │  (pub/sub telemetry)│ │             │
│  │  │  sidecars)     │  └─────────────────────┘ │             │
│  │  └────────────────┘                          │             │
│  │  ┌────────────────────────────────────────┐  │             │
│  │  │       Node Identity                    │  │             │
│  │  │  (fingerprint + cluster stubs)         │  │             │
│  │  └────────────────────────────────────────┘  │             │
│  └──────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

```
Inbound request
     │
     ▼
1. Cognitive Envelope Parser
   └─ Extracts: intent, coherence_score, governance, temporal, resources
   └─ Plain requests → default safe values (backward compat preserved)
     │
     ▼
2. Attention Allocator
   └─ Computes salience from: intent + governance_mode + coherence + budget
   └─ Under overload: denies low-salience requests with 503
   └─ Slots request into active queue
     │
     ▼
3. Constitutional Cloud Governance
   └─ Checks 7 constitutional laws (aligned with Niblit modules/constitutional_layer.py)
   └─ Cloud guards: token ceiling, recursion depth, prompt injection, lockdown mode
   └─ Strict mode (default): blocks on violation → 403
   └─ Permissive mode: logs violation, allows through
     │
     ▼
4. Model Orchestrator
   └─ Routes based on: governance_mode, coherence, resource_mode, trust scores
   └─ survival/lockdown → fastest model
   └─ minimal resource → fastest model
   └─ low coherence → fastest model
   └─ normal → highest composite score (trust × latency EMA)
     │
     ▼
5. GGUF Inference
   └─ Executes via llama-cpp-python
   └─ On failure: orchestrator records outcome (trust decay)
     │
     ▼
6. Post-processing
   └─ Orchestrator outcome recorded (trust, latency EMA)
   └─ Reflection engine turn recorded (quality, coherence, latency)
   └─ Temporal sync coherence updated
   └─ Attention slot released
     │
     ▼
Response
```

## Module Reference

### `app/event_bus.py` — Structured Event Bus
- Thread-safe pub/sub (aligned with Niblit `modules/event_bus.py`)
- Event type constants for all Ω.7 signals
- `CloudEvent` dataclass: `type`, `source`, `payload`, `timestamp`
- Singleton: `get_event_bus()`

### `app/cognitive_envelope.py` — Request Envelope
- Normalizes schema-v2 cognitive execution envelopes (niblit-lean-algos compatible)
- All fields optional — plain requests receive safe defaults
- Validates: intent, execution_mode, coherence_score, governance, temporal, resources
- `normalize_envelope(raw) → dict`

### `app/cloud_governance.py` — Constitutional Governance
- Enforces Niblit's 7 constitutional laws
- Additional cloud guards: token limit, recursion depth, prompt injection, lockdown, influence saturation
- Strict mode (default): hard block on violation
- Emits `EVENT_GOVERNANCE_CHECKED` / `EVENT_GOVERNANCE_VETOED`
- `GovernanceVerdict.to_dict()` exposes `allowed`, `violated`, `authority`, `rationale`

### `app/temporal_sync.py` — Temporal Coherence
- Tracks epoch ID, sync status, coherence EMA, drift detection
- Epoch tag parsing from request envelopes
- Sync states: `unsynced` → `syncing` → `synced` → `drifted`
- `EpochState.to_dict()` exposes all temporal fields

### `app/reflection_engine.py` — Cognitive Telemetry
- Records per-request: quality, latency, coherence, model, intent, veto, token count
- Rolling EMAs for quality, latency, coherence
- Governance veto rate tracking
- Hallucination risk proxy (low coherence + high token count)
- JSONL persistence (`NIBLIT_CLOUD_REFLECTION_FILE`)
- Auto-reflection every N requests (`NIBLIT_RE_CADENCE`)
- `ReflectionSnapshot.to_dict()` exposes all telemetry

### `app/model_orchestrator.py` — Multi-Model Routing
- `ModelHealth` per model: trust score, latency EMA, failure rate, consecutive failures
- Composite score: `trust_weight × trust - latency_weight × latency_penalty`
- Routing rules: governance mode → resource mode → coherence → composite score
- Fallback chain automatically constructed
- `record_outcome()` updates trust score (decay on failure, recovery on success)
- Emits `EVENT_ROUTING_DECISION`, `EVENT_MODEL_FAILURE`, `EVENT_MODEL_FALLBACK`

### `app/attention_allocator.py` — Attention Economy
- Salience scoring: intent weight × governance multiplier + coherence boost × budget
- Starvation prevention via aging factor (salience increases with wait time)
- Overload detection: pressure EMA ≥ 0.85 → deny low-salience requests
- Emergency survival mode: requests with salience < 0.5 denied under overload
- Emits `EVENT_ATTENTION_ALLOCATED`

### `app/trading_runtime_bridge.py` — Trading Cognition
- Reads schema-v2 signal from `NIBLIT_SIGNAL_FILE` (niblit-lean-algos compatible)
- Incremental JSONL sidecar ingestion: `NIBLIT_REFLECTION_FILE`, `NIBLIT_EPISODES_FILE`
- `TradingState`: signal, regime, governance_mode, forecast_consensus, volatility
- `inference_scale` property: scales down inference capacity under volatile/survival conditions
- Emits `EVENT_TRADE_REFLECTION_INGESTED`, `EVENT_MARKET_EPISODE_INGESTED`

### `app/node_identity.py` — Cluster Readiness
- Stable `node_id` from hostname + pid hash (overridable via `NIBLIT_NODE_ID`)
- `NodeCapabilities` advertisement for future federation
- `cluster_status()` returns single-node status (federation not yet implemented)
- Emits `EVENT_NODE_IDENTITY_SET` on init

### `app/federation.py` — Federation Preparation Stubs
- Defines portable federation interfaces without full distributed execution:
  - node registration
  - cluster discovery
  - capability advertisement
  - runtime heartbeat
  - governance synchronization
  - temporal epoch synchronization
- In-memory stub manager (`FederationManager`) preserves interface contracts for future rollout
- Exposed via `/federation/*` endpoints and integrated into `/cluster/status`

## Runtime Diagnostics Surface

`GET /v1/runtime/diagnostics` aggregates governance-aware operational telemetry:

- runtime health
- inference pressure / attention pressure
- model latency EMA map
- governance violation counters
- thermal/resource adaptation state
- reflection statistics
- coherence drift
- federation status (stub)

## Constitutional Laws (aligned with Niblit)

| Law | Condition that triggers violation |
|---|---|
| LAW_1: preserve_system_integrity | `stability_score < 0.3` OR `runtime_health < 0.3` |
| LAW_4: constrain_low_confidence_autonomy | `autonomous=True AND confidence < 0.35 AND coherence < 0.5` |
| LAW_6: temporal_incoherence_halts_execution | `coherence_score < 0.1` OR `epoch_alignment = "incoherent"` |
| LAW_7: safety_overrides_efficiency | `constitutional_priority = "safety" AND constitution_passed = False` |

Plus cloud guards:
- Token limit exceeded (`> NIBLIT_CG_MAX_TOKENS`)
- Recursion depth exceeded (`> NIBLIT_CG_MAX_RECURSION`)
- Prompt injection detected (heuristic pattern matching)
- Lockdown mode active
- Influence saturation (attention_available outside 0.0–1.0)

## Governance Modes

Aligned with niblit-lean-algos `TradeGovernanceGate` and Niblit lean_algo_manager:

| Mode | Description | Routing effect |
|---|---|---|
| `normal` | Default execution | Composite score routing |
| `cautious` | Reduced risk | Composite score routing (trust threshold raised) |
| `survival` | Minimal risk | Force fastest model |
| `lockdown` | No execution | Request blocked by governance |

## Event Bus Constants (Ω.7 Alignment)

Cloud runtime events mirror Niblit Phase Ω.7 naming:

| Cloud event | Niblit equivalent |
|---|---|
| `execution_envelope.published` | `EVENT_EXECUTION_ENVELOPE_PUBLISHED` |
| `trade_reflection.ingested` | `EVENT_TRADE_REFLECTION_INGESTED` |
| `market_episode.ingested` | `EVENT_MARKET_EPISODE_INGESTED` |
| `runtime_mode.changed` | `EVENT_RUNTIME_MODE_CHANGED` |
| `reflection.complete` | `EVENT_REFLECTION_COMPLETE` |
| `attention.allocated` | `EVENT_ATTENTION_ALLOCATED` |
| `coherence.evaluated` | `EVENT_COHERENCE_EVALUATED` |
| `constitution.checked` | `EVENT_CONSTITUTION_CHECKED` |

## Distributed Runtime Roadmap

Phase Ω.7 prepares architecture for future distributed cognition without implementing federation:

- ✅ Node identity + fingerprinting
- ✅ Capability advertisement
- ✅ `/cluster/status`, `/cluster/identity`, `/cluster/capabilities` endpoints
- ✅ `/federation/status`, `/federation/peers`, `/federation/register`, `/federation/discover`, `/federation/heartbeat`, `/federation/governance/sync`, `/federation/epoch/sync`
- ⬜ Node-to-node trust protocol
- ⬜ Distributed signal aggregation
- ⬜ Swarm consensus layer
- ⬜ Federated model routing
- ⬜ Distributed reflection synchronization

## Portable Runtime Tooling Layer

Operational tooling in `tools/` supports remote orchestration across cloud and edge:

- `cloud_runtime_ctl.py` (CLI) + `lib/runtime_client.py` (reusable API client)
- `install_runtime.sh` portable installer with version pinning and integrity checks
- `start_server.sh` lifecycle and smoke tooling for runtime operators

## Cognitive Gateway Layer

The Cognitive Gateway sits between external OpenAI-compatible clients (e.g. Cursor)
and the existing `/v1` inference handlers. It extends the Cursor Gateway Adapter
without modifying `ModelManager`, `GGUFEngine`, or `handle_chat()`.

```
Cursor / OpenAI client
        │
        ▼
┌───────────────────────────────────────┐
│  Cursor Gateway (app/cursor_gateway)  │
│  OpenAI normalization + SSE headers   │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│  Cognitive Gateway                      │
│  (app/cognitive_gateway)                │
│  ┌─────────────────────────────────┐  │
│  │ Request Classifier               │  │
│  │ coding / reasoning / memory /    │  │
│  │ tool_usage / general_chat        │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                      │
│  ┌─────────────────────────────────┐  │
│  │ Rule-based Model Router          │  │
│  │ dynamic / passthrough / static   │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                      │
│  ┌─────────────────────────────────┐  │
│  │ Hook System (structural)         │  │
│  │ memory before/after (placeholder)│ │
│  │ sync event bus (request-driven)  │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                      │
│  ┌─────────────────────────────────┐  │
│  │ Observability                    │  │
│  │ classification, routing, latency │  │
│  └─────────────────────────────────┘  │
└───────────────────┬───────────────────┘
                    ▼
        Existing /v1/chat/completions
        handle_chat() → ModelManager → GGUF
```

### Request flow

1. Cursor Gateway normalizes message shapes (`developer` → `system`, multimodal content).
2. Cognitive Gateway classifies the request and assigns compute priority.
3. Router selects a model using `ROUTING_STRATEGY` and routing rules from env/file.
4. Memory hook `before_request()` runs (no-op unless `MEMORY_HOOK_ENABLED=1`).
5. Existing `handle_chat()` performs inference unchanged.
6. Response is stripped of non-OpenAI fields; memory `after_response()` and events fire.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `ENABLE_COGNITIVE_GATEWAY` | `1` | Enable classification + routing layer |
| `ROUTING_STRATEGY` | `dynamic` | `dynamic`, `passthrough`, or `static` |
| `MEMORY_HOOK_ENABLED` | `0` | Placeholder memory injection hook |
| `TOOL_HOOK_ENABLED` | `0` | Placeholder tool/event hook gate |
| `COGNITIVE_GATEWAY_ROUTING_JSON` | `{}` | Request-type → model ID map |
| `COGNITIVE_GATEWAY_FALLBACK_MODEL` | — | Fallback when rule target unavailable |

### Resource safety

- No background threads, polling loops, or persistent async workers.
- Event bus handlers run synchronously per request.
- `GatewayContext` is request-scoped only.
