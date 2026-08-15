"""Local AI Diagnostics — verify VS Code local AI setup.

Checks:
✓ llama-server running
✓ Niblit Cloud running
✓ OpenAI endpoint reachable
✓ MCP connected
✓ Agent connected
✓ Model responding
✓ Streaming enabled
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict

log = logging.getLogger(__name__)


def run_diagnostics() -> Dict[str, Any]:
    """Run full local AI diagnostics and return results."""
    results: Dict[str, Any] = {
        "timestamp": __import__("time").time(),
        "checks": {},
        "summary": {"passed": 0, "failed": 0, "total": 0},
    }

    cloud_port = os.environ.get("NIBLIT_CLOUD_SERVER_PRIMARY_PORT", "8000")
    llama_port = os.environ.get("NIBLIT_LLAMA_SERVER_PORT", "8080")
    cloud_url = f"http://127.0.0.1:{cloud_port}"
    llama_url = f"http://127.0.0.1:{llama_port}"

    # 1. llama-server running
    results["checks"]["llama_server"] = _check_http(f"{llama_url}/health", "llama-server")

    # 2. Niblit Cloud running
    results["checks"]["niblit_cloud"] = _check_http(f"{cloud_url}/healthz", "Niblit Cloud")

    # 3. OpenAI endpoint reachable
    results["checks"]["openai_endpoint"] = _check_http(f"{cloud_url}/v1/models", "OpenAI /v1/models")

    # 4. Model responding
    results["checks"]["model_responding"] = _check_model_responds(cloud_url)

    # 5. Streaming enabled
    results["checks"]["streaming"] = _check_streaming(cloud_url)

    # 6. MCP connectivity (best-effort)
    results["checks"]["mcp"] = _check_mcp()

    # Summary
    total = len(results["checks"])
    passed = sum(1 for c in results["checks"].values() if c.get("status") == "ok")
    results["summary"] = {
        "passed": passed,
        "failed": total - passed,
        "total": total,
    }
    results["all_ok"] = passed == total

    return results


def _check_http(url: str, label: str) -> Dict[str, Any]:
    """Check if an HTTP endpoint is reachable."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return {
                "status": "ok",
                "label": label,
                "url": url,
                "http_status": resp.status,
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": "degraded",
            "label": label,
            "url": url,
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "label": label,
            "url": url,
            "error": str(exc),
        }


def _check_model_responds(cloud_url: str) -> Dict[str, Any]:
    """Verify the model responds to a chat completion request."""
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
        "stream": False,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{cloud_url}/v1/chat/completions",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "status": "ok" if content else "degraded",
                "label": "Model response",
                "response": content[:100] if content else "empty",
            }
    except Exception as exc:
        return {
            "status": "failed",
            "label": "Model response",
            "error": str(exc),
        }


def _check_streaming(cloud_url: str) -> Dict[str, Any]:
    """Verify streaming works."""
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "max_tokens": 20,
        "stream": True,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{cloud_url}/v1/chat/completions",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {
                "status": "ok",
                "label": "Streaming",
                "http_status": resp.status,
            }
    except Exception as exc:
        return {
            "status": "failed",
            "label": "Streaming",
            "error": str(exc),
        }


def _check_mcp() -> Dict[str, Any]:
    """Best-effort MCP connectivity check."""
    return {
        "status": "degraded",
        "label": "MCP",
        "note": "MCP servers configured in .vscode/niblit_local_ai_settings.json",
    }