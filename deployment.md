# Niblit Cognitive Cloud Runtime — Deployment Guide

## Docker

```bash
# Build
docker build -t niblit-cloud-runtime .

# Run with a single GGUF model
docker run -p 8000:8000 \
  -e GGUF_MODELS_JSON='{"qwen2.5-0.5b":"/models/qwen2.5-0.5b.gguf"}' \
  -e DEFAULT_MODEL_ID=qwen2.5-0.5b \
  -v /your/model/dir:/models \
  niblit-cloud-runtime

# Run with cognitive runtime features
docker run -p 8000:8000 \
  -e GGUF_MODELS_JSON='{"qwen2.5-0.5b":"/models/qwen2.5-0.5b.gguf","llama3":"/models/llama3.gguf"}' \
  -e DEFAULT_MODEL_ID=qwen2.5-0.5b \
  -e NIBLIT_CG_STRICT=1 \
  -e NIBLIT_RE_ENABLED=1 \
  -e NIBLIT_AA_MAX_QUEUE=32 \
  -v /your/model/dir:/models \
  niblit-cloud-runtime
```

## Fly.io

```bash
fly launch --name niblit-cloud-runtime
fly secrets set GGUF_MODELS_JSON='{"qwen2.5-0.5b":"/app/models/qwen2.5-0.5b.gguf"}'
fly secrets set DEFAULT_MODEL_ID=qwen2.5-0.5b
fly secrets set NIBLIT_CG_STRICT=1
fly deploy
```

See `fly.toml` for machine configuration.

## Portable Runtime Toolkit (Cloud / VPS / ARM / Edge)

The `tools/` directory provides portable operational tooling:

```bash
# Install runtime backend (llama-server default)
./tools/install_runtime.sh local-linux llama-server

# Alternative targets
./tools/install_runtime.sh flyio llama-server
./tools/install_runtime.sh docker llama-cli
./tools/install_runtime.sh arm-server llama-server
./tools/install_runtime.sh edge-arm llama-cli
```

Version pinning and integrity:

```bash
export LLAMA_CPP_VERSION=b5380
export LLAMA_SERVER_URL="https://example.com/llama-server"
export LLAMA_SERVER_SHA256="<sha256>"
./tools/install_runtime.sh vps llama-server
```

Runtime operations:

```bash
./tools/start_server.sh start
./tools/start_server.sh status
./tools/start_server.sh smoke
./tools/start_server.sh stop
```

Runtime control:

```bash
python tools/cloud_runtime_ctl.py --url http://127.0.0.1:8000 status
python tools/cloud_runtime_ctl.py --url http://127.0.0.1:8000 diagnostics
python tools/cloud_runtime_ctl.py --socket /tmp/niblit-runtime.sock health
python tools/cloud_runtime_ctl.py --tcp-host 127.0.0.1 --tcp-port 9009 cluster
python tools/niblit_ctl.py --url http://127.0.0.1:8000 topology
python tools/niblit_ctl.py --url http://127.0.0.1:8000 compatibility
```

Runtime profiles:

```bash
source tools/runtime_profiles/profile_loader.sh cloud-server
source tools/runtime_profiles/profile_loader.sh edge-runtime
source tools/runtime_profiles/profile_loader.sh degraded-runtime
source tools/runtime_profiles/profile_loader.sh disconnected-runtime
```

## HuggingFace Spaces

1. Set repository type to **Docker**.
2. Mount your GGUF model files.
3. Set environment secrets for `GGUF_MODELS_JSON` and `DEFAULT_MODEL_ID`.

## Environment Variables Reference

### Core inference

| Variable | Default | Required | Description |
|---|---|---|---|
| `GGUF_MODELS_JSON` | `{}` | Yes | JSON map: `model_id → gguf_path` |
| `DEFAULT_MODEL_ID` | first model | No | Default model for alias requests |
| `COMPAT_PREFIXES` | `hf,local,kimi,claude` | No | Compatibility URL prefixes |
| `N_CTX` | `4096` | No | llama.cpp context length |
| `N_THREADS` | `4` | No | llama.cpp thread count |

### Model Orchestrator

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_MO_ENABLED` | `1` | Enable intelligent model routing |
| `NIBLIT_MO_LATENCY_WEIGHT` | `0.3` | Latency penalty weight in routing score |
| `NIBLIT_MO_TRUST_WEIGHT` | `0.7` | Trust score weight in routing score |
| `NIBLIT_CLOUD_LLM_URL` | `` | Remote Ollama/LMStudio URL (optional) |

### Constitutional Governance

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_CG_ENABLED` | `1` | Enable constitutional governance layer |
| `NIBLIT_CG_STRICT` | `1` | `1` = block violations; `0` = log only |
| `NIBLIT_CG_MAX_TOKENS` | `8192` | Hard token ceiling per request |
| `NIBLIT_CG_MAX_RECURSION` | `16` | Max recursion depth |

### Temporal Coherence

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_TS_ENABLED` | `1` | Enable epoch tracking |
| `NIBLIT_TS_EMA_ALPHA` | `0.1` | EMA smoothing factor for coherence |

### Reflection Engine

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_RE_ENABLED` | `1` | Enable reflection telemetry |
| `NIBLIT_RE_CADENCE` | `50` | Auto-reflection every N requests |
| `NIBLIT_CLOUD_REFLECTION_FILE` | `/tmp/niblit_cloud_reflection.jsonl` | JSONL output path |

### Attention Economy

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_AA_ENABLED` | `1` | Enable salience-based prioritization |
| `NIBLIT_AA_MAX_QUEUE` | `64` | Max concurrent active requests |
| `NIBLIT_AA_OVERLOAD_RATIO` | `0.85` | Overload trigger threshold |

### Trading Bridge

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_TRADING_BRIDGE_ENABLED` | `1` | Enable trading cognition integration |
| `NIBLIT_SIGNAL_FILE` | `/tmp/niblit_lean_signal.json` | Schema-v2 signal envelope path |
| `NIBLIT_SIGNAL_MAX_AGE` | `300` | Max signal age (seconds) before treating as stale |
| `NIBLIT_REFLECTION_FILE` | `/tmp/niblit_trade_reflection.jsonl` | Trade reflection sidecar |
| `NIBLIT_EPISODES_FILE` | `/tmp/niblit_market_episodes.jsonl` | Market episodes sidecar |

### Node Identity

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_NODE_ID` | auto-generated | Override stable node ID |
| `NIBLIT_NODE_REGION` | `local` | Geographic region hint |
| `NIBLIT_NODE_ROLE` | `inference` | Node role for future federation |

### Federation Preparation

| Variable | Default | Description |
|---|---|---|
| `NIBLIT_FEDERATION_ENABLED` | `0` | Enable federation stubs in status surfaces |
| `NIBLIT_FEDERATION_REGISTRY` | `` | Registry URL placeholder for future federation |
| `NIBLIT_FEDERATION_MAX_PEERS` | `8` | Max peers placeholder |
| `NIBLIT_HEARTBEAT_INTERVAL` | `30` | Heartbeat interval placeholder |

## Connecting niblit-lean-algos

When running niblit-lean-algos alongside this server, set the signal file
paths to shared volumes so the cloud runtime can read the trading cognition state:

```bash
# In niblit-lean-algos
NIBLIT_SIGNAL_FILE=/shared/niblit_lean_signal.json
NIBLIT_REFLECTION_FILE=/shared/niblit_trade_reflection.jsonl
NIBLIT_EPISODES_FILE=/shared/niblit_market_episodes.jsonl

# In cloud server
NIBLIT_SIGNAL_FILE=/shared/niblit_lean_signal.json
NIBLIT_REFLECTION_FILE=/shared/niblit_trade_reflection.jsonl
NIBLIT_EPISODES_FILE=/shared/niblit_market_episodes.jsonl
```

The cloud runtime reads these files on every `/v1/runtime/trading` call.

## Migration from Phase Ω.6

All existing deployments continue to work without changes.  To enable
Phase Ω.7 features:

1. No breaking changes to existing API contracts.
2. Optionally add cognitive envelope fields to requests.
3. Monitor `/v1/runtime/status` for health overview.
4. Set `NIBLIT_CG_STRICT=0` initially if you want permissive governance while
   calibrating coherence thresholds.
5. Check `/metrics/governance` for veto rate before enabling strict mode.

## Hybrid Edge/Phone/Cloud Topology (Preparation)

Recommended future topology (supported operationally by current tooling):

- phone/edge local runtime using `llama-cli` + UNIX/TCP control transport
- cloud runtime using HTTP admin mode and governance diagnostics
- shared governance semantics (Ω.7 cognitive envelope) across all nodes
- staged federation rollout via `/federation/*` interfaces (currently stubs)

## Running Tests

```bash
python -m pytest -q
```

All 155 tests should pass. Original backward-compatibility tests are preserved.
