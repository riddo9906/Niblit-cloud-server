# Local Llama.cpp Setup for Niblit Workspace

This document describes how to configure the Niblit workspace to use a local llama.cpp server as the default AI backend for all coding tasks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VS Code Workspace                             │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Cline / Continue Extensions                               │   │
│  │   → OpenAI-compatible API (http://127.0.0.1:8000/v1)     │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  llama.cpp Server (qwen_server.sh)                 │
│  Endpoint: http://127.0.0.1:8000                               │
│  Model: qwen2.5-coder-3b-instruct-q4_k_m.gguf                    │
│  Context: 32768 tokens                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Startup Procedure

### 1. Start the llama.cpp Server

```powershell
# From Git Bash or WSL
cd /c/Users/Riyaad/Documents/GitHub/Niblit-cloud-server/tools
./qwen_server.sh

# Or from PowerShell (requires Git Bash)
wsl ./qwen_server.sh

# Or manually:
C:/Users/Riyaad/llama_migration/llama.cpp/build/bin/Release/llama-server.exe `
    -m C:/Users/Riyaad/llama_migration/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf `
    --host 127.0.0.1 --port 8000 `
    -c 32768 -t 8 --n-gpu-layers 35 `
    --cont-batching --batch-size 128 --ubatch-size 64
```

### 2. Verify Server Status

```powershell
python tools/check_local_llm.py
```

Expected output: All checks should pass with ✅

### 3. Configure Environment

```powershell
# Copy the local LLM environment template
cp .env.local_llm .env.local_llm.local

# Source in your shell (Git Bash)
source .env.local_llm.local
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOCAL_MODEL_ENABLED` | Enable local model as primary backend | `true` |
| `LOCAL_MODEL_PROVIDER` | Provider type | `llama_cpp` |
| `LOCAL_MODEL_PATH` | Path to GGUF model file | `C:/Users/Riyaad/llama_migration/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` |
| `LLAMA_CPP_ROOT` | llama.cpp installation directory | `C:/Users/Riyaad/llama_migration/llama.cpp` |
| `LOCAL_MODEL_SERVER_SCRIPT` | Path to qwen_server.sh | `C:/Users/Riyaad/Documents/GitHub/Niblit-cloud-server/tools/qwen_server.sh` |
| `LOCAL_MODEL_ENDPOINT` | HTTP endpoint | `http://127.0.0.1:8000` |
| `LOCAL_MODEL_CHAT_ENDPOINT` | Chat completions path | `/v1/chat/completions` |
| `LOCAL_MODEL_COMPLETION_ENDPOINT` | Completions path | `/v1/completions` |
| `LOCAL_MODEL_STREAM` | Enable streaming | `true` |
| `LOCAL_MODEL_DEFAULT_TEMPERATURE` | Default temperature | `0.7` |
| `LOCAL_MODEL_DEFAULT_MAX_TOKENS` | Default max tokens | `1024` |
| `LOCAL_MODEL_N_CTX` | Context window size | `32768` |
| `LOCAL_MODEL_N_THREADS` | Number of threads | `8` |
| `LOCAL_MODEL_N_GPU_LAYERS` | GPU layers (0 for CPU only) | `35` |

## Model Replacement

To swap the GGUF model:

1. Place your new model at `C:/Users/Riyaad/llama_migration/models/<model-name>.gguf`
2. Update `LOCAL_MODEL_PATH` in `.env.local_llm`
3. Restart the server with the new model:
   ```powershell
   NIBLIT_MODEL_PATH=C:/Users/Riyaad/llama_migration/models/<new-model>.gguf ./qwen_server.sh
   ```
4. The provider automatically detects the model from `/v1/models` - no code changes needed

## Troubleshooting

### Server Won't Start

```powershell
# Check if port is already in use
netstat -ano | findstr :8000

# Kill any existing llama-server process
taskkill /F /IM llama-server.exe

# Verify model path exists
ls -la C:/Users/Riyaad/llama_migration/models/
```

### VS Code Can't Connect

1. Verify server is running:
   ```powershell
   curl.exe http://127.0.0.1:8000/health
   ```

2. Check VS Code settings include:
   - `cline.openAiBaseUrl`: `http://127.0.0.1:8000/v1`
   - `cline.openAiModelId`: `local`

3. Restart VS Code after configuration changes

### Slow Responses

- Check latency with `python tools/check_local_llm.py`
- Adjust `N_THREADS` and `N_GPU_LAYERS` for your hardware
- Reduce `max_tokens` in requests

### Memory Issues

- Reduce `N_CTX` if running out of VRAM
- Lower `--n-gpu-layers` to use more CPU RAM
- Use a smaller quantization (Q5 or Q4 instead of Q8)

## Offline Workflow

1. Start the server before opening VS Code
2. Ensure no cloud API keys are set in environment
3. All LLM requests will route to local server
4. Server provides caching between requests (no reload)

## VS Code Integration

### Cline

Settings are pre-configured in `.vscode/settings.json`:
- Uses OpenAI-compatible endpoint
- Auto-detects model from `/v1/models`
- Supports streaming responses

### Continue

Configuration in `.continue/config.json`:
- Custom commands for explain, refactor, review, test
- System prompt optimized for coding

## Repository Understanding

The local model supports repository-aware operations through:

1. **Context injection**: Pass repository context in system messages
2. **Multi-file operations**: Include file content in prompts
3. **Code review**: Use `code_review()` method with repository files
4. **Architecture analysis**: Use `reasoning()` with codebase summary

Example usage:
```python
from tools.local_llm_provider import get_provider

provider = get_provider()
result = provider.repository_chat(
    messages=[{"role": "user", "content": "How should I structure this module?"}],
    context=repo_context
)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Context Window | 32768 tokens |
| Model Size | ~2GB (Q4_K_M) |
| GPU Layers | 35 (configurable) |
| Threads | 8 (configurable) |
| Average Latency | ~400ms (CPU) |
| Streaming | Supported |
| Keep-alive | Built-in (server stays loaded) |

## Priority Chain

When `LOCAL_MODEL_ENABLED=true`:

1. **Local llama.cpp** (primary) - Qwen2.5-Coder-3B
2. **Anthropic** (fallback) - If configured
3. **HuggingFace API** (fallback) - If configured
4. **OpenAI** (fallback) - If configured