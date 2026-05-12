#!/usr/bin/env python3
"""tools/lib/runtime_client.py — Niblit Cognitive Cloud Runtime HTTP Client.

Reusable client library for querying and controlling a running Niblit Cognitive
Cloud Runtime instance.  Used by ``tools/cloud_runtime_ctl.py`` and can be
imported directly into automation scripts.

Transport modes
---------------
- **HTTP / HTTPS** — default; works for Fly.io, Docker, remote servers.
- **TCP admin socket** — connects to a raw TCP admin listener (future).
- **UNIX domain socket** — connects via a local UNIX socket (local dev).

All methods are synchronous.  Import ``RuntimeClient`` and construct it with
the base URL of your runtime instance:

    from tools.lib.runtime_client import RuntimeClient
    client = RuntimeClient("https://niblit-cloud-server.fly.dev")
    print(client.health())
    print(client.runtime_status())
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

# Default timeout for HTTP requests (seconds)
DEFAULT_TIMEOUT: int = 15


@dataclass
class RuntimeResponse:
    """Result of a client request."""
    ok: bool
    status_code: int
    data: dict[str, Any]
    error: str = ""
    url: str = ""

    def __repr__(self) -> str:
        if self.ok:
            return f"<RuntimeResponse {self.status_code} ok url={self.url}>"
        return f"<RuntimeResponse {self.status_code} error={self.error!r} url={self.url}>"


class RuntimeClient:
    """HTTP client for the Niblit Cognitive Cloud Runtime admin API.

    Args:
        base_url:       Base URL of the runtime (e.g. ``http://localhost:8000``
                        or ``https://niblit-cloud-server.fly.dev``).
        timeout:        Request timeout in seconds.
        admin_token:    Optional bearer token for protected admin endpoints.
        unix_socket:    Path to UNIX domain socket (overrides HTTP transport).
        tcp_host:       Optional TCP admin host (JSON-over-TCP transport).
        tcp_port:       Optional TCP admin port.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = DEFAULT_TIMEOUT,
        admin_token: str = "",
        unix_socket: str = "",
        tcp_host: str = "",
        tcp_port: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.admin_token = admin_token
        self.unix_socket = unix_socket
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port

    # ── Probe endpoints ────────────────────────────────────────────────────────

    def health(self) -> RuntimeResponse:
        """GET /health — liveness probe."""
        return self._get("/health")

    def props(self) -> RuntimeResponse:
        """GET /props — llama-server compatibility probe."""
        return self._get("/props")

    def list_models(self) -> RuntimeResponse:
        """GET /v1/models — list registered models."""
        return self._get("/v1/models")

    # ── Runtime status ─────────────────────────────────────────────────────────

    def runtime_status(self) -> RuntimeResponse:
        """GET /v1/runtime/status — full runtime snapshot."""
        return self._get("/v1/runtime/status")

    def runtime_mode(self) -> RuntimeResponse:
        """GET /v1/runtime/mode — runtime mode + adaptation posture."""
        return self._get("/v1/runtime/mode")

    def coherence(self) -> RuntimeResponse:
        """GET /v1/runtime/coherence — temporal coherence state."""
        return self._get("/v1/runtime/coherence")

    def governance(self) -> RuntimeResponse:
        """GET /v1/runtime/governance — constitutional governance stats."""
        return self._get("/v1/runtime/governance")

    def attention(self) -> RuntimeResponse:
        """GET /v1/runtime/attention — attention economy metrics."""
        return self._get("/v1/runtime/attention")

    def models(self) -> RuntimeResponse:
        """GET /v1/runtime/models — model orchestration health."""
        return self._get("/v1/runtime/models")

    def reflection(self) -> RuntimeResponse:
        """GET /v1/runtime/reflection — reflection engine telemetry."""
        return self._get("/v1/runtime/reflection")

    def trading(self) -> RuntimeResponse:
        """GET /v1/runtime/trading — trading cognition bridge state."""
        return self._get("/v1/runtime/trading")

    def epoch(self) -> RuntimeResponse:
        """GET /v1/runtime/epoch — current epoch and coherence."""
        return self._get("/v1/runtime/epoch")

    # ── Metrics ────────────────────────────────────────────────────────────────

    def metrics_cognitive(self) -> RuntimeResponse:
        """GET /metrics/cognitive — cognitive telemetry."""
        return self._get("/metrics/cognitive")

    def metrics_coherence(self) -> RuntimeResponse:
        """GET /metrics/coherence — coherence metrics."""
        return self._get("/metrics/coherence")

    def metrics_governance(self) -> RuntimeResponse:
        """GET /metrics/governance — governance metrics."""
        return self._get("/metrics/governance")

    def metrics_models(self) -> RuntimeResponse:
        """GET /metrics/models — model health metrics."""
        return self._get("/metrics/models")

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def diagnostics(self) -> RuntimeResponse:
        """GET /v1/runtime/diagnostics — comprehensive operational diagnostics."""
        return self._get("/v1/runtime/diagnostics")

    # ── Cluster / Federation ───────────────────────────────────────────────────

    def cluster_status(self) -> RuntimeResponse:
        """GET /cluster/status — cluster status."""
        return self._get("/cluster/status")

    def cluster_identity(self) -> RuntimeResponse:
        """GET /cluster/identity — node identity."""
        return self._get("/cluster/identity")

    def node_identity(self) -> RuntimeResponse:
        """GET /v1/runtime/node — runtime node identity."""
        return self._get("/v1/runtime/node")

    def cluster_capabilities(self) -> RuntimeResponse:
        """GET /cluster/capabilities — node capabilities."""
        return self._get("/cluster/capabilities")

    def federation_peers(self) -> RuntimeResponse:
        """GET /federation/peers — known federation peers (stub)."""
        return self._get("/federation/peers")

    def federation_register(self, node_info: dict[str, Any]) -> RuntimeResponse:
        """POST /federation/register — register this node with a peer (stub)."""
        return self._post("/federation/register", node_info)

    # ── Inference ──────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "local",
        temperature: float = 0.2,
        max_tokens: int = 256,
        envelope: dict[str, Any] | None = None,
    ) -> RuntimeResponse:
        """POST /v1/chat/completions — run a chat inference."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if envelope:
            payload.update(envelope)
        return self._post("/v1/chat/completions", payload)

    def cognitive_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "local",
        intent: str = "conversational",
        coherence_score: float = 1.0,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> RuntimeResponse:
        """POST /v1/cognitive/chat — enriched cognitive chat."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "intent": intent,
            "coherence_score": coherence_score,
        }
        return self._post("/v1/cognitive/chat", payload)

    # ── Composite helpers ──────────────────────────────────────────────────────

    def full_status(self) -> dict[str, Any]:
        """Collect a comprehensive status snapshot from multiple endpoints.

        Returns a dict with all subsystem statuses, marking errors where
        individual endpoints were unreachable.
        """
        results: dict[str, Any] = {}
        for name, method in [
            ("health", self.health),
            ("runtime", self.runtime_status),
            ("runtime_mode", self.runtime_mode),
            ("coherence", self.coherence),
            ("governance", self.governance),
            ("attention", self.attention),
            ("models", self.models),
            ("reflection", self.reflection),
            ("trading", self.trading),
            ("epoch", self.epoch),
            ("cluster", self.cluster_status),
        ]:
            resp = method()
            results[name] = resp.data if resp.ok else {"error": resp.error}
        return results

    def is_healthy(self) -> bool:
        """Return True if /health returns 200."""
        return self.health().ok

    # ── Transport ──────────────────────────────────────────────────────────────

    def _get(self, path: str) -> RuntimeResponse:
        return self._request("GET", path, None)

    def _post(self, path: str, body: dict[str, Any]) -> RuntimeResponse:
        return self._request("POST", path, body)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> RuntimeResponse:
        if self.unix_socket:
            return self._unix_request(method, path, body)
        if self.tcp_host and self.tcp_port > 0:
            return self._tcp_request(method, path, body)
        return self._http_request(method, path, body)

    def _http_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> RuntimeResponse:
        url = self.base_url + path
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.admin_token:
            headers["Authorization"] = f"Bearer {self.admin_token}"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode()
                return RuntimeResponse(
                    ok=True,
                    status_code=resp.status,
                    data=json.loads(raw) if raw else {},
                    url=url,
                )
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode()
                err_data = json.loads(raw) if raw else {}
            except Exception:
                err_data = {}
            return RuntimeResponse(
                ok=False,
                status_code=exc.code,
                data=err_data,
                error=str(exc),
                url=url,
            )
        except Exception as exc:
            return RuntimeResponse(
                ok=False,
                status_code=0,
                data={},
                error=str(exc),
                url=url,
            )

    def _unix_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> RuntimeResponse:
        """Minimal HTTP/1.0 request over a UNIX domain socket."""
        url = f"unix:{self.unix_socket}{path}"
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.unix_socket)

            body_bytes = json.dumps(body).encode() if body else b""
            content_type = "application/json"
            request_line = f"{method} {path} HTTP/1.0\r\n"
            headers = (
                f"Host: localhost\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
            )
            if self.admin_token:
                headers += f"Authorization: Bearer {self.admin_token}\r\n"
            raw_request = (request_line + headers + "\r\n").encode() + body_bytes
            sock.sendall(raw_request)

            # Read response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
            sock.close()

            # Parse HTTP response
            header_end = response_data.find(b"\r\n\r\n")
            header_part = response_data[:header_end].decode()
            body_part = response_data[header_end + 4:]
            status_line = header_part.split("\r\n")[0]
            status_code = int(status_line.split(" ")[1])
            parsed_body = json.loads(body_part) if body_part else {}
            return RuntimeResponse(
                ok=(200 <= status_code < 300),
                status_code=status_code,
                data=parsed_body,
                url=url,
            )
        except Exception as exc:
            return RuntimeResponse(
                ok=False, status_code=0, data={}, error=str(exc), url=url
            )

    def _tcp_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> RuntimeResponse:
        """JSON-over-TCP admin request transport."""
        url = f"tcp://{self.tcp_host}:{self.tcp_port}{path}"
        payload = {
            "method": method,
            "path": path,
            "body": body or {},
            "token": self.admin_token,
        }
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.tcp_host, self.tcp_port))
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            raw = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                raw += chunk
            sock.close()
            text = raw.decode("utf-8").strip()
            data = json.loads(text) if text else {}
            status_code = int(data.get("status_code", 200))
            return RuntimeResponse(
                ok=(200 <= status_code < 300),
                status_code=status_code,
                data=data.get("data", data),
                error=str(data.get("error", "")),
                url=url,
            )
        except Exception as exc:
            return RuntimeResponse(
                ok=False, status_code=0, data={}, error=str(exc), url=url
            )
