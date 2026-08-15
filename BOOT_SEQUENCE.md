# Niblit Cloud Server — Boot Sequence

## Canonical entrypoint

```bash
python main.py
```

This executes `_run_cli()` from `app.main`, which runs the layered `CloudRuntime` boot sequence and then starts Uvicorn.

## Boot layers

```
Layer 1  CONFIGURATION
  Load GGUF_MODELS_JSON, DEFAULT_MODEL_ID, n_ctx, n_threads, n_batch, n_ubatch

Layer 2  STORAGE
  Create NIBLIT_STORAGE_ROOT (default: .niblit/storage)

Layer 3  MODEL_REGISTRY
  Register all available model IDs

Layer 4  INFERENCE
  Seed inference config (n_ctx, n_threads, n_batch, n_ubatch)

Layer 4b EMBEDDINGS
  Best-effort initialization of embedding service

Layer 5  VECTOR_DATABASE
  Best-effort initialization of vector store

Layer 6  API_SERVICES
  Initialize cognitive subsystems:
    - EventBus
    - CognitiveEnvelope
    - NodeIdentity
    - CloudGovernance
    - TemporalSync
    - AttentionAllocator
    - ReflectionEngine
    - ModelOrchestrator
    - TradingRuntimeBridge
    - FederationManager

Layer 7  WEBSOCKET
  Deferred (not yet implemented)

Layer 8  SCHEDULER
  Best-effort startup of scheduler/workers if available

Layer 9  BACKGROUND_SERVICES
  Start background workers:
    - Scheduler
    - WorkerManager

Layer 9b PLUGINS
  Load plugins from NIBLIT_PLUGINS_DIR if configured

Layer 10 HEALTH
  Initialize ModelManager
  Verify health of all subsystems

READY / DEGRADED
```

## Console output

Expected output during boot:

```
Cloud Server Boot
[CONFIGURATION] COMPLETED (X.X ms): models=N default=model-id
[STORAGE] COMPLETED (X.X ms): /path/to/.niblit/storage
[MODEL_REGISTRY] COMPLETED (X.X ms): registered=N
[INFERENCE] COMPLETED (X.X ms): n_ctx=16384 n_threads=4 ...
[EMBEDDINGS] COMPLETED/DEGRADED (X.X ms): ...
[VECTOR_DATABASE] COMPLETED/DEGRADED (X.X ms): ...
[API_SERVICES] COMPLETED (X.X ms): 0.0.0.0:8000
[WEBSOCKET] DEGRADED (X.X ms): websocket_deferred
[SCHEDULER] DEGRADED (X.X ms): scheduler_deferred
[BACKGROUND_SERVICES] COMPLETED/DEGRADED (X.X ms): ...
[PLUGINS] COMPLETED/DEGRADED (X.X ms): ...
[HEALTH] COMPLETED (X.X ms): health_verified

Cloud READY
```

## Graceful shutdown

When the server receives SIGINT/SIGTERM, the runtime calls all registered shutdown callbacks in reverse order.

## Preserving qwen_server.sh

`tools/qwen_server.sh` is retained as a backend helper for launching llama-server directly. It is no longer the primary entrypoint. If needed, the runtime can invoke it for model serving; currently the Python runtime uses llama-cpp-python directly.

## Dependency graph

```
CONFIGURATION
  └─ STORAGE
       └─ MODEL_REGISTRY
            └─ INFERENCE
                 ├─ EMBEDDINGS
                 └─ VECTOR_DATABASE
                      └─ API_SERVICES
                           ├─ EventBus
                           ├─ CognitiveEnvelope
                           ├─ NodeIdentity
                           ├─ CloudGovernance
                           ├─ TemporalSync
                           ├─ AttentionAllocator
                           ├─ ReflectionEngine
                           ├─ ModelOrchestrator
                           ├─ TradingRuntimeBridge
                           └─ FederationManager
                                ├─ WEBSOCKET
                                ├─ SCHEDULER
                                ├─ BACKGROUND_SERVICES
                                │    ├─ Scheduler
                                │    └─ WorkerManager
                                ├─ PLUGINS
                                └─ HEALTH
                                     └─ ModelManager