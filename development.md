# Niblit Cognitive Cloud Server — Developer Guide

## Development Environment Setup

```bash
# Clone and install
git clone https://github.com/riddo9906/Niblit-cloud-server.git
cd Niblit-cloud-server
pip install -r requirements.txt
```

### Run tests

```bash
python -m pytest -q
# 155 tests expected to pass
```

### Start the server locally

```bash
uvicorn app.main:app --reload --port 8000
# or via helper:
./tools/start_server.sh start
./tools/start_server.sh status
./tools/start_server.sh smoke
./tools/start_server.sh stop
```

---

## Runtime Profile System

Runtime profiles centralize deployment configuration and prevent environment drift.

### Available profiles

| Profile | Description | Resource class |
|---|---|---|
| `cloud-server` | Cloud server (Fly.io, Docker, VPS) | balanced |
| `niblit` | Niblit main-app runtime | balanced |
| `termux-local` | Termux / ARM edge device | minimal |
| `local-runtime` | Portable local runtime topology | balanced |
| `edge-runtime` | Portable edge runtime topology | minimal |
| `degraded-runtime` | Degraded/high-pressure runtime | degraded |
| `disconnected-runtime` | Offline/disconnected deterministic runtime | isolated |

### Using profiles (shell)

```bash
# Activate a profile in current shell
source tools/runtime_profiles/profile_loader.sh cloud-server
source tools/runtime_profiles/profile_loader.sh termux-local

# Or via environment variable
NIBLIT_PROFILE=termux-local source tools/runtime_profiles/profile_loader.sh
```

### Using profiles (Python)

```python
from tools.lib.runtime_profiles import load_profile, get_profile_env, profile_summary

# Load into os.environ (safe merge — existing vars are not overwritten)
load_profile("cloud-server")

# Load with override
load_profile("termux-local", override=True)

# Get env dict without modifying os.environ
env = get_profile_env("termux-local")

# Summary with governance metadata
summary = profile_summary("cloud-server")
# {"runtime_mode": "normal", "governance_strict": True, "resource_class": "balanced", ...}
```

### Profile structure

Profiles are `.env` files in `tools/runtime_profiles/`:

- `niblit.env` — Niblit main-app runtime
- `cloud-server.env` — Cloud server defaults
- `termux-local.env` — Edge / Termux runtime
- `local-runtime.env` — Portable local runtime
- `edge-runtime.env` — Portable edge runtime
- `degraded-runtime.env` — Survival-mode degraded runtime
- `disconnected-runtime.env` — Lockdown-mode disconnected runtime

Each profile defines: server address, socket paths, model paths, governance defaults,
tunnel provider, federation stubs, and reflection telemetry paths.

---

## Runtime Control CLI

`tools/cloud_runtime_ctl.py` — governance-aware CLI for operating a running server.
`tools/niblit_ctl.py` — thin wrapper on sidecar client for transport-normalized control.

```bash
# Basic commands
python tools/cloud_runtime_ctl.py --url http://localhost:8000 ping
python tools/cloud_runtime_ctl.py --url http://localhost:8000 health
python tools/cloud_runtime_ctl.py --url http://localhost:8000 status
python tools/cloud_runtime_ctl.py --url http://localhost:8000 diagnostics
python tools/cloud_runtime_ctl.py --url http://localhost:8000 governance
python tools/cloud_runtime_ctl.py --url http://localhost:8000 coherence
python tools/cloud_runtime_ctl.py --url http://localhost:8000 reflection
python tools/cloud_runtime_ctl.py --url http://localhost:8000 cluster

# Watch mode (continuous poll)
python tools/cloud_runtime_ctl.py --url http://localhost:8000 watch 5

# Send test message
python tools/cloud_runtime_ctl.py --url http://localhost:8000 chat "hello"

# UNIX socket transport
python tools/cloud_runtime_ctl.py --socket /tmp/niblit-cloud-server.sock health

# TCP admin transport
python tools/cloud_runtime_ctl.py --tcp-host 127.0.0.1 --tcp-port 9009 health

# Raw JSON output for piping
python tools/cloud_runtime_ctl.py --url http://localhost:8000 --json diagnostics

# sidecar wrapper
python tools/niblit_ctl.py --url http://localhost:8000 status
python tools/niblit_ctl.py --socket /tmp/niblit-cloud-server.sock federation
```

---

## Sidecar Client Library

`tools/lib/sidecar_client.py` — reusable low-level IPC client for Niblit components.

```python
from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig, from_env, normalize_envelope

# Create from environment variables
client = from_env()

# Create with explicit config
client = SidecarClient(SidecarClientConfig(
    http_base_url="http://localhost:8000",
    unix_socket="",
    tcp_host="",
    tcp_port=0,
    token="",
    output_mode="pretty",
))

# Standard requests
resp = client.health()
resp = client.diagnostics()
resp = client.governance()

# Chat with schema-v2 envelope
resp = client.chat(
    messages=[{"role": "user", "content": "hello"}],
    envelope={
        "intent": "conversational",
        "coherence_score": 0.95,
        "governance": {"governance_mode": "normal"},
    }
)

# Normalize envelope (schema-v2 alignment)
safe_envelope = normalize_envelope({"intent": "trading", "coherence_score": 1.5})
# {"intent": "trading", "coherence_score": 1.0, ...}

# Streaming (HTTP only)
for token in client.stream_chat(messages=[{"role": "user", "content": "hi"}]):
    print(token, end="", flush=True)
```

### Schema-v2 envelope normalization

The sidecar client normalizes cognitive envelopes aligned with Ω.7 schema-v2:

- Unknown `intent` values → `"conversational"`
- `coherence_score` clamped to [0.0, 1.0]
- `attention_budget` clamped to [0.0, 1.0]
- Unknown `governance_mode` → `"normal"`
- Unknown fields are preserved

Canonical governance modes: `normal`, `cautious`, `survival`, `lockdown`, `minimal`

---

## Termux / Edge Runtime

### Quick start on Termux / ARM

```bash
# 1. Install llama.cpp backend
./tools/install_llama_server.sh --backend llama-server

# 2. Launch inference server
# termux-local auto-resolves ~/models/*.gguf and ~/llama.cpp/build/bin/llama-server
./tools/termux_inference_server.sh --profile termux-local

# 3. With cloudflared tunnel
./tools/termux_inference_server.sh \
  --model ~/models/qwen2.5-1.5b.gguf \
  --profile termux-local \
  --tunnel cloudflared

# 4. With manual public URL (e.g. from existing tunnel)
NIBLIT_TUNNEL_PUBLIC_URL=https://my-tunnel.example.com \
  ./tools/termux_inference_server.sh --model ~/models/model.gguf

# 5. Dry run (validate config without starting)
./tools/termux_inference_server.sh --model ~/models/model.gguf --dry-run
```

### Termux runtime launcher options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | `$NIBLIT_MODEL_PATH` | GGUF model path |
| `--port PORT` | `8000` | Bind port |
| `--n-ctx N` | `2048` | Context window |
| `--n-threads N` | `4` | CPU threads |
| `--n-gpu-layers N` | `0` | GPU offload layers |
| `--tunnel PROVIDER` | `none` | `cloudflared`, `ngrok`, `none` |
| `--public-url URL` | | Manual public URL override |
| `--profile PROFILE` | | Load a runtime profile |
| `--no-governance` | | Disable governance telemetry log |
| `--dry-run` | | Print config without starting |

### Set Niblit to use the Termux runtime

```bash
# In Niblit environment
export NIBLIT_LLAMA_SERVER_URL=https://your-tunnel-url
export NIBLIT_GGUF_BACKEND=http
```

---

## Local Runtime Validator

`tools/install_local_runtime.py` — platform detection and runtime validation.

```bash
# Full validation (platform, backend, model, Niblit integration)
python tools/install_local_runtime.py --validate

# Platform info
python tools/install_local_runtime.py --info

# Validate a specific model
python tools/install_local_runtime.py --check-model ~/models/model.gguf

# Check backend binary
python tools/install_local_runtime.py --check-backend

# Install instructions for current platform
python tools/install_local_runtime.py --install-hints

# Probe running server
python tools/install_local_runtime.py --probe --port 8000
```

---

## llama.cpp Installer

`tools/install_llama_server.sh` — portable llama.cpp backend installer.

```bash
# Install llama-server (build from source if no URL provided)
./tools/install_llama_server.sh

# Install specific version
./tools/install_llama_server.sh --version b5380

# Install to custom directory
./tools/install_llama_server.sh --target /usr/local/bin

# Download prebuilt binary (with checksum verification)
LLAMA_SERVER_URL="https://example.com/llama-server" \
LLAMA_SERVER_SHA256="abc123..." \
./tools/install_llama_server.sh

# Install both llama-server and llama-cli
./tools/install_llama_server.sh --backend both

# Skip if already installed
./tools/install_llama_server.sh --skip-if-exists

# Force reinstall
./tools/install_llama_server.sh --force

# Dry run
./tools/install_llama_server.sh --dry-run
```

---

## Architecture and Module Map

### Core server

| Module | Description |
|---|---|
| `app/main.py` | FastAPI application, routing, model management, all endpoints |
| `app/cognitive_envelope.py` | Schema-v2 cognitive envelope (Ω.7 alignment) |
| `app/cloud_governance.py` | Constitutional governance (token/recursion limits, veto) |
| `app/temporal_sync.py` | Temporal coherence tracking and epoch management |
| `app/reflection_engine.py` | Quality reflection, EMA latency, auto-reflection cadence |
| `app/model_orchestrator.py` | Multi-model routing, trust scoring, load tracking |
| `app/attention_allocator.py` | Request queue + overload pressure management |
| `app/trading_runtime_bridge.py` | niblit-lean-algos signal/regime/coherence bridge |
| `app/node_identity.py` | Stable node ID, capabilities, cluster/swarm readiness |
| `app/federation.py` | Federation preparation stubs (interfaces reserved for future) |
| `app/event_bus.py` | Structured runtime event bus (Ω.7 event constants) |

### Tooling

| File | Description |
|---|---|
| `tools/cloud_runtime_ctl.py` | Governance-aware runtime control CLI |
| `tools/lib/runtime_client.py` | HTTP runtime client (used by cloud_runtime_ctl.py) |
| `tools/lib/sidecar_client.py` | Low-level sidecar IPC client (UNIX/TCP/HTTP, schema-v2) |
| `tools/lib/runtime_profiles.py` | Runtime profile loader (Python API) |
| `tools/runtime_profiles/*.env` | Profile definitions |
| `tools/runtime_profiles/profile_loader.sh` | Shell profile loader |
| `tools/termux_inference_server.sh` | Termux/ARM inference server launcher |
| `tools/install_llama_server.sh` | Portable llama.cpp installer |
| `tools/install_local_runtime.py` | Multi-platform local runtime validator |
| `tools/install_runtime.sh` | Legacy portable installer (preserved) |
| `tools/start_server.sh` | Server start/stop/status/smoke helper |

---

## Testing

```bash
# Full suite
python -m pytest -q

# Specific suites
python -m pytest tests/test_api.py -q           # Backward compat (13 tests)
python -m pytest tests/test_cognitive_runtime.py -q  # Cognitive runtime (92 tests)
python -m pytest tests/test_runtime_tooling_layer.py -q  # Tooling layer (63 tests)

# Targeted
python -m pytest tests/test_runtime_tooling_layer.py -k "profile" -q
python -m pytest tests/test_runtime_tooling_layer.py -k "sidecar" -q
python -m pytest tests/test_runtime_tooling_layer.py -k "backward" -q
```

---

## GitHub Workflow Architecture

| Workflow | Trigger | Purpose |
|---|---|---|
| `test.yml` | push/PR to main | Full test suite (Python 3.11, 3.12) |
| `deploy.yml` | push/PR to main | Pre-deployment validation, module imports, API compatibility |
| `runtime_tooling_validation.yml` | push/PR (tools/ or app/ changes) | Runtime profile + sidecar + tooling layer validation |
| `anti_drift.yml` | push/PR (app/ or tools/ changes) | Schema/governance/naming drift detection |
| `portability.yml` | push/PR (tools/ changes) | Portability and platform-safety checks |

### Workflow philosophy

All workflows use:
- `concurrency:` groups with `cancel-in-progress: true` for PRs
- `timeout-minutes:` to prevent runaway jobs
- `fail-fast: false` on matrix strategies
- Python 3.11 + 3.12 matrix coverage
- Artifact uploads for test results

### Anti-drift protection

The `anti_drift.yml` workflow validates:
- Governance mode naming alignment (Ω.7 constants)
- Schema-v2 envelope field normalization
- Protocol version constants
- Profile completeness
- Cloud-only assumption detection (advisory)

---

## Ecosystem Integration

### Niblit integration

```bash
# Point Niblit's QwenLocalBrain at this server
export NIBLIT_LLAMA_SERVER_URL=https://your-cloud-server-url
export NIBLIT_GGUF_BACKEND=http
```

Niblit's `QwenLocalBrain._generate_http()` always sends `"model": "local"` to the
configured `NIBLIT_LLAMA_SERVER_URL/v1/chat/completions`. This server handles it as
the default model alias.

### niblit-lean-algos integration

The trading cognition bridge (`app/trading_runtime_bridge.py`) reads schema-v2 signal
envelopes from `$NIBLIT_SIGNAL_FILE` and exposes them via `/v1/runtime/trading`.

Schema-v2 signal fields: `signal`, `confidence`, `regime`, `coherence_score`,
`governance_mode`, `attention_budget`, `epoch_id`, `intent`.

---

## Contributing

1. Run `python -m pytest -q` before committing — all 155 tests must pass
2. New tooling must target at minimum: `local-linux`, `termux-local`, `cloud-server` profiles
3. Governance mode names must match `GOVERNANCE_MODES` in `tools/lib/sidecar_client.py`
4. Schema-v2 envelope field additions must be normalized in `normalize_envelope()`
5. New shell scripts must use `set -euo pipefail` and cleanup traps
