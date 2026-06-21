#!/usr/bin/env python3
"""Niblit runtime sidecar client with topology and compatibility metadata."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Generator

SIDECAR_PROTOCOL_VERSION = "sidecar/1.1"
DEFAULT_TIMEOUT = 15
CHUNK_SIZE = 4096

CANONICAL_RUNTIME_MODES = frozenset({"normal", "cautious", "survival", "lockdown"})
_MODE_ALIASES = {"constrained": "cautious", "minimal": "cautious"}
GOVERNANCE_MODES = frozenset(set(CANONICAL_RUNTIME_MODES) | set(_MODE_ALIASES.keys()))

INTENT_TYPES = frozenset(
    {
        "trading",
        "forecasting",
        "reasoning",
        "analytical",
        "tool_use",
        "code_generation",
        "summarization",
        "conversational",
        "creative",
        "unknown",
    }
)

COMPATIBILITY_METADATA = {
    "schema_version": "2.x",
    "event_contract_version": "omega-7",
    "governance_contract_version": "1.x",
    "advisor_protocol_version": "2.x",
    "runtime_mode_contract": "2026.05",
}


@dataclass
class SidecarRequestContext:
    """Replay-safe request metadata for runtime lineage and topology."""

    runtime_profile: str = "cloud-server"
    runtime_mode: str = "normal"
    topology: str = "cloud"
    replay_id: str = ""
    lineage_id: str = ""

    def normalized(self) -> dict[str, Any]:
        mode = normalize_runtime_mode(self.runtime_mode)
        return {
            "runtime_profile": self.runtime_profile,
            "runtime_mode": mode,
            "governance_mode": mode,
            "topology": self.topology,
            "replay_id": self.replay_id,
            "lineage_id": self.lineage_id,
            "protocol": SIDECAR_PROTOCOL_VERSION,
            "compatibility": dict(COMPATIBILITY_METADATA),
        }


@dataclass
class SidecarResponse:
    ok: bool
    status_code: int
    data: dict[str, Any]
    error: str = ""
    url: str = ""
    latency_ms: float = 0.0
    protocol: str = SIDECAR_PROTOCOL_VERSION
    transport: str = "unknown"
    compatibility: dict[str, Any] | None = None

    def __repr__(self) -> str:
        if self.ok:
            return (
                f"<SidecarResponse {self.status_code} ok "
                f"transport={self.transport} latency={self.latency_ms:.1f}ms>"
            )
        return f"<SidecarResponse {self.status_code} error={self.error!r}>"


@dataclass
class SidecarClientConfig:
    unix_socket: str = ""
    tcp_host: str = ""
    tcp_port: int = 0
    http_base_url: str = "http://localhost:8000"
    token: str = ""
    timeout: int = DEFAULT_TIMEOUT
    output_mode: str = "pretty"  # pretty | json | raw


def normalize_runtime_mode(mode: object, default: str = "normal") -> str:
    candidate = str(mode or default).strip().lower()
    candidate = _MODE_ALIASES.get(candidate, candidate)
    if candidate not in CANONICAL_RUNTIME_MODES:
        return default
    return candidate


def normalize_envelope(envelope: dict[str, Any] | None) -> dict[str, Any]:
    if not envelope or not isinstance(envelope, dict):
        return {}

    normalized = dict(envelope)

    intent = str(normalized.get("intent", "conversational")).lower()
    if intent not in INTENT_TYPES:
        intent = "conversational"
    normalized["intent"] = intent

    try:
        normalized["coherence_score"] = max(0.0, min(1.0, float(normalized.get("coherence_score", 1.0))))
    except (TypeError, ValueError):
        normalized["coherence_score"] = 1.0

    try:
        normalized["attention_budget"] = max(0.0, min(1.0, float(normalized.get("attention_budget", 1.0))))
    except (TypeError, ValueError):
        normalized["attention_budget"] = 1.0

    gov = normalized.get("governance")
    if isinstance(gov, dict):
        gov = dict(gov)
        gov["governance_mode"] = normalize_runtime_mode(gov.get("governance_mode", "normal"))
        normalized["governance"] = gov

    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime = dict(runtime)
        runtime["mode"] = normalize_runtime_mode(runtime.get("mode", "normal"))
        normalized["runtime"] = runtime

    return normalized


def check_compatibility(payload: dict[str, Any] | None) -> dict[str, Any]:
    incoming = dict(payload or {}).get("compatibility") or {}
    if not isinstance(incoming, dict):
        incoming = {}

    mismatches: dict[str, dict[str, str]] = {}
    for key, expected in COMPATIBILITY_METADATA.items():
        got = str(incoming.get(key, "")).strip()
        if got and got != expected:
            mismatches[key] = {"expected": expected, "received": got}

    return {
        "compatible": len(mismatches) == 0,
        "expected": dict(COMPATIBILITY_METADATA),
        "received": incoming,
        "mismatches": mismatches,
    }


class SidecarClient:
    def __init__(self, config: SidecarClientConfig | None = None) -> None:
        self._cfg = config or SidecarClientConfig()
        self._context = SidecarRequestContext()

    def set_context(self, context: SidecarRequestContext) -> None:
        self._context = context

    def health(self) -> SidecarResponse:
        return self.get("/health")

    def runtime_status(self) -> SidecarResponse:
        return self.get("/v1/runtime/status")

    def diagnostics(self) -> SidecarResponse:
        return self.get("/v1/runtime/diagnostics")

    def governance(self) -> SidecarResponse:
        return self.get("/v1/runtime/governance")

    def coherence(self) -> SidecarResponse:
        return self.get("/v1/runtime/coherence")

    def federation_status(self) -> SidecarResponse:
        return self.get("/federation/status")

    def topology(self) -> SidecarResponse:
        response = self.get("/v1/runtime/topology")
        if response.ok:
            return response
        return self.get("/cluster/status")

    def compatibility(self) -> SidecarResponse:
        payload = {
            "compatibility": COMPATIBILITY_METADATA,
            "context": self._context.normalized(),
        }
        return self.post("/federation/governance/sync", payload)

    def active_model(self) -> SidecarResponse:
        """Return the currently active model and all registered models."""
        return self.get("/v1/runtime/model/active")

    def switch_model(self, model_id: str) -> SidecarResponse:
        """Switch the active model to *model_id* while the server is running."""
        return self.post("/v1/runtime/model/switch", {"model_id": model_id})

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "local",
        temperature: float = 0.2,
        max_tokens: int = 256,
        envelope: dict[str, Any] | None = None,
    ) -> SidecarResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if envelope:
            payload.update(normalize_envelope(envelope))
        return self.post("/v1/chat/completions", payload)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "local",
        temperature: float = 0.2,
        max_tokens: int = 256,
        envelope: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if envelope:
            payload.update(normalize_envelope(envelope))
        payload["_runtime_coord"] = self._context.normalized()

        url = self._cfg.http_base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"

        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        return
                    try:
                        obj = json.loads(chunk)
                        token = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except Exception as exc:  # noqa: BLE001
            yield f"[stream error: {exc}]"

    def get(self, path: str) -> SidecarResponse:
        return self._request("GET", path, None)

    def post(self, path: str, body: dict[str, Any]) -> SidecarResponse:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> SidecarResponse:
        t0 = time.perf_counter()
        attempts: list[tuple[str, Any]] = []

        if self._cfg.unix_socket:
            attempts.append(("unix", self._unix_request))
        if self._cfg.tcp_host and self._cfg.tcp_port > 0:
            attempts.append(("tcp", self._tcp_request))
        if not attempts:
            attempts.append(("http", self._http_request))

        payload = dict(body or {}) if body else None
        if payload is not None:
            payload.setdefault("_runtime_coord", self._context.normalized())

        last = SidecarResponse(ok=False, status_code=0, data={}, error="no transport attempted")
        for transport, request_fn in attempts:
            resp = request_fn(method, path, payload)
            resp.transport = transport
            resp.compatibility = check_compatibility(resp.data)
            if resp.ok or resp.status_code > 0:
                resp.latency_ms = (time.perf_counter() - t0) * 1000
                return resp
            last = resp

        last.latency_ms = (time.perf_counter() - t0) * 1000
        return last

    def _url_for(self, path: str) -> str:
        return self._cfg.http_base_url.rstrip("/") + path

    def _http_request(self, method: str, path: str, body: dict[str, Any] | None) -> SidecarResponse:
        url = self._url_for(path)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                parsed = _safe_json(raw)
                return SidecarResponse(ok=True, status_code=resp.status, data=parsed, url=url)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return SidecarResponse(
                ok=False,
                status_code=exc.code,
                data=_safe_json(raw),
                error=str(exc),
                url=url,
            )
        except Exception as exc:  # noqa: BLE001
            return SidecarResponse(ok=False, status_code=0, data={}, error=str(exc), url=url)

    def _unix_request(self, method: str, path: str, body: dict[str, Any] | None) -> SidecarResponse:
        url = f"unix:{self._cfg.unix_socket}{path}"
        if not hasattr(socket, "AF_UNIX"):
            return SidecarResponse(ok=False, status_code=0, data={}, error="unix sockets unsupported on this platform", url=url)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._cfg.timeout)
        try:
            sock.connect(self._cfg.unix_socket)
            body_bytes = json.dumps(body).encode() if body else b""
            headers = (
                "Host: localhost\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
            )
            if self._cfg.token:
                headers += f"Authorization: Bearer {self._cfg.token}\r\n"
            request = (f"{method} {path} HTTP/1.0\r\n" + headers + "\r\n").encode() + body_bytes
            sock.sendall(request)

            response_data = b""
            while True:
                chunk = sock.recv(CHUNK_SIZE)
                if not chunk:
                    break
                response_data += chunk

            idx = response_data.find(b"\r\n\r\n")
            if idx < 0:
                return SidecarResponse(ok=False, status_code=0, data={}, error="malformed_http_response", url=url)

            header = response_data[:idx].decode("utf-8", errors="replace")
            body_text = response_data[idx + 4 :].decode("utf-8", errors="replace")
            status_line = header.split("\r\n", 1)[0]
            status_code = int(status_line.split(" ")[1])
            return SidecarResponse(
                ok=200 <= status_code < 300,
                status_code=status_code,
                data=_safe_json(body_text),
                url=url,
            )
        except Exception as exc:  # noqa: BLE001
            return SidecarResponse(ok=False, status_code=0, data={}, error=str(exc), url=url)
        finally:
            sock.close()

    def _tcp_request(self, method: str, path: str, body: dict[str, Any] | None) -> SidecarResponse:
        url = f"tcp://{self._cfg.tcp_host}:{self._cfg.tcp_port}{path}"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._cfg.timeout)

        payload = {
            "method": method,
            "path": path,
            "body": body or {},
            "token": self._cfg.token,
            "protocol": SIDECAR_PROTOCOL_VERSION,
            "compatibility": COMPATIBILITY_METADATA,
        }
        try:
            sock.connect((self._cfg.tcp_host, self._cfg.tcp_port))
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

            buffer = b""
            while True:
                chunk = sock.recv(CHUNK_SIZE)
                if not chunk:
                    break
                buffer += chunk
                if b"\n" in buffer:
                    break

            line = buffer.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
            data = _safe_json(line)
            status_code = int(data.get("status_code", 200))
            return SidecarResponse(
                ok=200 <= status_code < 300,
                status_code=status_code,
                data=data.get("data", data),
                error=str(data.get("error", "")),
                url=url,
            )
        except Exception as exc:  # noqa: BLE001
            return SidecarResponse(ok=False, status_code=0, data={}, error=str(exc), url=url)
        finally:
            sock.close()

    def render(self, resp: SidecarResponse, mode: str | None = None) -> str:
        effective = mode or self._cfg.output_mode
        if effective == "json":
            return json.dumps(resp.data, separators=(",", ":"), default=str)
        if effective == "raw":
            return str(resp.data)
        return json.dumps(resp.data, indent=2, default=str)


def _safe_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": raw, "malformed": True}


def from_env() -> SidecarClient:
    import os

    client = SidecarClient(
        SidecarClientConfig(
            unix_socket=os.getenv("NIBLIT_UNIX_SOCKET", ""),
            tcp_host=os.getenv("NIBLIT_TCP_ADMIN_HOST", ""),
            tcp_port=int(os.getenv("NIBLIT_TCP_ADMIN_PORT", "0")),
            http_base_url=os.getenv("NIBLIT_CLOUD_URL", "http://localhost:8000"),
            token=os.getenv("NIBLIT_ADMIN_TOKEN", ""),
            timeout=int(os.getenv("NIBLIT_TIMEOUT", str(DEFAULT_TIMEOUT))),
        )
    )
    client.set_context(
        SidecarRequestContext(
            runtime_profile=os.getenv("NIBLIT_PROFILE", "cloud-server"),
            runtime_mode=normalize_runtime_mode(os.getenv("NIBLIT_RUNTIME_MODE", "normal")),
            topology=os.getenv("NIBLIT_RUNTIME_TOPOLOGY", "cloud"),
            replay_id=os.getenv("NIBLIT_REPLAY_ID", ""),
            lineage_id=os.getenv("NIBLIT_RUNTIME_LINEAGE_ID", ""),
        )
    )
    return client
