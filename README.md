# Niblit Cloud Server (GGUF + Hugging Face-style API)

Standalone Python FastAPI server that serves local GGUF models through
Hugging Face-compatible chat/inference endpoints so it can be used directly by
Niblit provider logic and other compatible clients.

## Features

- Hugging Face-style chat completions endpoint (`/v1/chat/completions`)
- Hugging Face-style inference endpoint (`/models/{model}`)
- Multiple compatibility routes for provider URL patterns:
  - `/hf/...`
  - `/local/...`
  - `/kimi/...`
  - `/claude/...`
- Model discovery endpoints:
  - `GET /v1/models`
  - `GET /v1/models/{model}`
- GGUF runtime via `llama-cpp-python`
- Ready for Docker / Fly.io / Hugging Face Spaces deployment

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

- `GGUF_MODELS_JSON` — JSON map of `model_id -> gguf_path`
  - Example:
    `{"qwen2.5-0.5b":"./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"}`
- `DEFAULT_MODEL_ID` — model id used if request does not provide one
- `COMPAT_PREFIXES` — comma-separated compatibility URL prefixes
  (default: `hf,local,kimi,claude`)
- `N_CTX` — context length for llama.cpp (default: `4096`)
- `N_THREADS` — thread count for llama.cpp (default: `4`)

## API examples

### Chat completions (HF-style)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model":"qwen2.5-0.5b",
    "messages":[{"role":"user","content":"Hello from Niblit"}],
    "temperature":0.2,
    "max_tokens":128
  }'
```

### Inference API format

```bash
curl -X POST http://localhost:8000/models/qwen2.5-0.5b \
  -H "content-type: application/json" \
  -d '{"inputs":"Hello from Niblit"}'
```

## Compatibility notes

- The server exposes several endpoint aliases so clients configured for
  `hf`, `local`, `kimi`, or `claude` provider-style base URLs can call the same
  backend without custom rewrite logic.
- If you need extra vendor mapping, set `COMPAT_PREFIXES` and keep your client
  pointed at the same deployment base URL.
