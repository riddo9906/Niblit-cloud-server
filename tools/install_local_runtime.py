#!/usr/bin/env python3
"""tools/install_local_runtime.py — Niblit portable local runtime validator.

Multi-platform validator and setup helper for local GGUF inference backends.
Validates that the runtime environment is correctly configured for running
Niblit cognitive inference locally on Termux, Linux, Docker, or other platforms.

Usage
-----
    python tools/install_local_runtime.py [--validate | --info | --check-model MODEL]

    # Validate full runtime stack (filesystem, backend, model)
    python tools/install_local_runtime.py --validate

    # Print platform info and environment
    python tools/install_local_runtime.py --info

    # Check a specific model file
    python tools/install_local_runtime.py --check-model /path/to/model.gguf

    # Validate llama-server binary only
    python tools/install_local_runtime.py --check-backend

    # Install instructions for current platform
    python tools/install_local_runtime.py --install-hints

Platforms
---------
- Termux (Android)
- Linux (x86_64, arm64)
- Docker / container environments

Model families supported
------------------------
- Qwen (qwen2.5, qwen3, etc.)
- Llama (llama3.x, llama2)
- Mistral / Mixtral
- Phi / Phi-3
- Gemma
- Any GGUF-compatible model
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

LLAMA_SERVER_BINS = ("llama-server", "llamafile", "llama.cpp/llama-server")
KNOWN_MODEL_FAMILIES = ("qwen", "llama", "mistral", "phi", "gemma", "mixtral", "deepseek")
HEALTH_PROBE_TIMEOUT = 10
DEFAULT_PORT = 8000


# ── Platform detection ────────────────────────────────────────────────────────


def detect_platform() -> dict[str, Any]:
    """Detect the current runtime platform and capabilities."""
    uname = platform.uname()
    is_termux = "com.termux" in os.environ.get("PREFIX", "") or \
                "/data/data/com.termux" in os.environ.get("HOME", "")
    is_container = Path("/.dockerenv").exists() or \
                   bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    is_linux = uname.system == "Linux"
    is_arm = uname.machine in ("aarch64", "arm64", "armv7l", "armv7")
    is_x86 = uname.machine in ("x86_64", "amd64")

    return {
        "os": uname.system,
        "arch": uname.machine,
        "python_version": platform.python_version(),
        "is_termux": is_termux,
        "is_container": is_container,
        "is_linux": is_linux,
        "is_arm": is_arm,
        "is_x86": is_x86,
        "cpu_count": os.cpu_count() or 1,
        "hostname": uname.node,
        "prefix": os.environ.get("PREFIX", ""),
        "home": str(Path.home()),
    }


def detect_model_family(model_path: str) -> str:
    """Detect the model family from a GGUF model filename."""
    name = Path(model_path).name.lower()
    for family in KNOWN_MODEL_FAMILIES:
        if family in name:
            return family
    return "unknown"


# ── Backend validation ────────────────────────────────────────────────────────


def find_llama_server() -> str | None:
    """Find the llama-server binary in PATH or known locations."""
    for binary in LLAMA_SERVER_BINS:
        found = shutil.which(binary)
        if found:
            return found
    # Check common install locations
    extra_dirs = [
        Path.home() / ".local" / "niblit-runtime" / "bin",
        Path("/usr/local/bin"),
        Path("/data/data/com.termux/files/usr/bin"),  # Termux default
    ]
    for d in extra_dirs:
        for binary in LLAMA_SERVER_BINS:
            candidate = d / binary
            if candidate.exists() and os.access(str(candidate), os.X_OK):
                return str(candidate)
    return None


def validate_backend(backend_path: str | None = None) -> dict[str, Any]:
    """Validate the llama-server backend."""
    path = backend_path or find_llama_server()
    if not path:
        return {
            "ok": False,
            "error": "llama-server not found in PATH or common locations",
            "hints": [
                "Run: ./tools/install_llama_server.sh --backend llama-server",
                "Or: pkg install llama-cpp (Termux)",
            ],
        }
    return {
        "ok": True,
        "binary": path,
        "in_path": shutil.which(Path(path).name) == path,
    }


# ── Model validation ──────────────────────────────────────────────────────────


def validate_model(model_path: str) -> dict[str, Any]:
    """Validate a GGUF model file exists and has reasonable size."""
    p = Path(model_path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {model_path}"}
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {model_path}"}

    size_bytes = p.stat().st_size
    size_gb = size_bytes / (1024 ** 3)

    # Check GGUF magic bytes
    is_gguf = False
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            is_gguf = magic == b"GGUF"
    except OSError:
        pass

    family = detect_model_family(model_path)

    result: dict[str, Any] = {
        "ok": True,
        "path": model_path,
        "size_bytes": size_bytes,
        "size_gb": round(size_gb, 2),
        "is_gguf": is_gguf,
        "family": family,
        "warnings": [],
    }

    if not is_gguf:
        result["warnings"].append("file may not be a valid GGUF model (magic bytes mismatch)")
    if size_bytes < 100 * 1024:
        result["warnings"].append("file is very small — may be a test/stub file")

    return result


def find_models_in_dir(model_dir: str | None = None) -> list[str]:
    """Find GGUF model files in a directory."""
    search_dirs = []
    if model_dir:
        search_dirs.append(Path(model_dir))
    # Common locations
    search_dirs += [
        Path.home() / "models",
        Path.home() / ".niblit" / "models",
        Path("/data/data/com.termux/files/home/models"),  # Termux
        Path("/opt/models"),
    ]

    found = []
    for d in search_dirs:
        if d.is_dir():
            for f in d.rglob("*.gguf"):
                found.append(str(f))
    return found


# ── Environment validation ────────────────────────────────────────────────────


def validate_env() -> dict[str, Any]:
    """Validate runtime environment variables."""
    required: list[tuple[str, str]] = [
        # (var_name, description) — all are optional but desirable
    ]
    recommended = [
        ("NIBLIT_MODEL_PATH", "path to GGUF model file"),
        ("NIBLIT_LLAMA_SERVER_URL", "URL where llama-server will listen"),
        ("NIBLIT_GGUF_BACKEND", "backend type: llama-server|http|stub"),
    ]

    present = {k: os.environ.get(k, "") for k, _ in recommended}
    missing = [k for k, _ in recommended if not present.get(k)]

    return {
        "ok": True,
        "env_vars": present,
        "missing_recommended": missing,
        "note": "All listed vars are optional but improve runtime portability",
    }


# ── Health probe ──────────────────────────────────────────────────────────────


def probe_runtime_health(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> dict[str, Any]:
    """Probe a running llama-server instance."""
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            body = resp.read().decode()
            return {
                "ok": True,
                "url": url,
                "status_code": resp.status,
                "latency_ms": round(latency_ms, 1),
                "body": body[:200],
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": str(exc)}


# ── QwenLocalBrain integration check ─────────────────────────────────────────


def check_qwen_integration() -> dict[str, Any]:
    """Check if QwenLocalBrain can connect to a running runtime.

    This is a filesystem-only check when the Niblit repo is not available.
    When Niblit is importable, attempts a real integration check.
    """
    niblit_url = os.environ.get("NIBLIT_LLAMA_SERVER_URL", "http://127.0.0.1:8000")
    result: dict[str, Any] = {
        "niblit_url": niblit_url,
        "niblit_importable": False,
        "probe": {},
    }

    try:
        # Try importing QwenLocalBrain from Niblit if available
        sys.path.insert(0, os.environ.get("NIBLIT_REPO_PATH", ""))
        from modules.local_brain import QwenLocalBrain  # type: ignore[import-not-found]  # noqa: F401
        result["niblit_importable"] = True
        result["note"] = "QwenLocalBrain importable — set NIBLIT_GGUF_BACKEND=http to use this runtime"
    except ImportError:
        result["note"] = (
            "QwenLocalBrain not importable (Niblit not in PYTHONPATH).  "
            "Set NIBLIT_LLAMA_SERVER_URL + NIBLIT_GGUF_BACKEND=http in Niblit."
        )

    # Probe the runtime regardless
    result["probe"] = probe_runtime_health()
    return result


# ── Install hints ─────────────────────────────────────────────────────────────


def install_hints(pf: dict[str, Any]) -> list[str]:
    """Return platform-appropriate setup instructions."""
    hints = []
    if pf["is_termux"]:
        hints += [
            "# Termux setup",
            "pkg update && pkg upgrade -y",
            "pkg install -y python cmake clang wget curl",
            "pip install fastapi uvicorn",
            "# Install llama.cpp:",
            "tools/install_llama_server.sh --backend llama-server",
            "# Or via pkg:",
            "pkg install llama-cpp",
        ]
    elif pf["is_linux"] and pf["is_arm"]:
        hints += [
            "# ARM Linux setup",
            "sudo apt-get install -y python3 python3-pip cmake gcc g++",
            "pip3 install fastapi uvicorn",
            "tools/install_llama_server.sh --backend llama-server",
        ]
    elif pf["is_container"]:
        hints += [
            "# Container setup (handled via Dockerfile)",
            "pip install -r requirements.txt",
            "# llama-server is typically pre-installed in container image",
        ]
    else:
        hints += [
            "# Linux/macOS setup",
            "pip install -r requirements.txt",
            "tools/install_llama_server.sh --backend llama-server",
            "# Optional: set INSTALL_DIR to customize install location",
        ]
    return hints


# ── Full validate ─────────────────────────────────────────────────────────────


def run_full_validate(args: argparse.Namespace) -> int:
    """Run full validation and print results."""
    pf = detect_platform()
    model_path = args.model or os.environ.get("NIBLIT_MODEL_PATH", "")

    results: dict[str, Any] = {
        "platform": pf,
        "backend": validate_backend(),
        "environment": validate_env(),
    }

    if model_path:
        results["model"] = validate_model(model_path)
    else:
        found = find_models_in_dir()
        results["model_search"] = {
            "searched": True,
            "found_count": len(found),
            "models": found[:5],
            "note": "set NIBLIT_MODEL_PATH to a .gguf file for inference",
        }

    results["qwen_integration"] = check_qwen_integration()

    print(json.dumps(results, indent=2, default=str))

    # Determine exit code
    ok = results["backend"]["ok"]
    if model_path:
        ok = ok and results.get("model", {}).get("ok", False)
    return 0 if ok else 1


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="install_local_runtime",
        description="Niblit portable local runtime validator",
    )
    parser.add_argument("--validate", action="store_true", help="Run full validation (default)")
    parser.add_argument("--info", action="store_true", help="Print platform info")
    parser.add_argument("--check-backend", action="store_true", help="Validate llama-server binary")
    parser.add_argument("--check-model", metavar="MODEL", help="Validate a specific model file")
    parser.add_argument("--model", default="", help="Model path (also $NIBLIT_MODEL_PATH)")
    parser.add_argument("--install-hints", action="store_true", help="Print setup instructions")
    parser.add_argument("--probe", action="store_true", help="Probe running llama-server health")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for health probe")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.info:
        pf = detect_platform()
        print(json.dumps(pf, indent=2, default=str))
        return 0

    if args.check_backend:
        result = validate_backend()
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    if args.check_model:
        result = validate_model(args.check_model)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    if args.install_hints:
        pf = detect_platform()
        for hint in install_hints(pf):
            print(hint)
        return 0

    if args.probe:
        result = probe_runtime_health(port=args.port)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    # Default: full validate
    return run_full_validate(args)


if __name__ == "__main__":
    sys.exit(main())
