#!/usr/bin/env python3
"""tools/lib/sidecar_client.py — Niblit Sidecar Protocol Client.

Provides a lightweight client for communicating with Niblit sidecar processes
via UNIX domain sockets, TCP, or HTTP.

The sidecar protocol is a newline-delimited JSON request/response protocol
used for low-latency local IPC between Niblit components.  This client
normalizes transport differences and provides schema-safe parsing aligned
with schema-v2 cognitive envelopes.

Protocol
--------
Request (newline-delimited JSON):
    {"method": "GET", "path": "/health", "body": {}, "token": ""}

Response (newline-delimited JSON):
    {"status_code": 200, "data": {...}, "error": ""}

Transport modes
---------------
- **UNIX socket** — fastest, for same-machine IPC (Termux, local Linux)
- **TCP socket**  — for cross-process or cross-network admin (Docker, VPS)
- **HTTP**        — standard HTTP/1.1 for cloud and remote instances

Output modes
------------
- ``pretty``  — indented JSON (default)
- ``json``    — compact JSON (for piping)
- ``raw``     — raw response bytes as string

Streaming
---------
Streaming responses are handled via chunked line-buffered reads.
The ``stream_chat`` method yields token strings incrementally.

Schema-v2 alignment
-------------------
- Cognitive envelope fields are preserved in request bodies.
- Governance mode, intent, coherence_score are normalized on input.
- Malformed envelopes are handled gracefully (best-effort normalization).

Configuration constants
-----------------------
- ``SIDECAR_PROTOCOL_VERSION`` — current protocol version string
- ``DEFAULT_TIMEOUT``          — default socket/HTTP timeout (seconds)
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Generator

# ── Protocol constants ────────────────────────────────────────────────────────

SIDECAR_PROTOCOL_VERSION = "sidecar/1.0"
DEFAULT_TIMEOUT: int = 15
CHUNK_SIZE: int = 4096

# Canonical governance modes (aligned with Ω.7)
GOVERNANCE_MODES = frozenset({"normal", "cautious", "survival", "lockdown", "minimal"})

# Schema-v2 intent types
INTENT_TYPES = frozenset({
    "trading", "forecasting", "reasoning", "analytical", "tool_use",
    "code_generation", "summarization", "conversational", "creative", "unknown",
})


# ── Response type ─────────────────────────────────────────────────────────────


@dataclass
class SidecarResponse:
    """Normalized response from a sidecar call."""
    ok: bool
    status_code: int
    data: dict[str, Any]
    error: str = ""
    url: str = ""
    latency_ms: float = 0.0
    protocol: str = SIDECAR_PROTOCOL_VERSION

    def __repr__(self) -> str:
        if self.ok:
            return f"<SidecarResponse {self.status_code} ok latency={self.latency_ms:.1f}ms>"
        return f"<SidecarResponse {self.status_code} error={self.error!r}>"


# ── Envelope normalization ────────────────────────────────────────────────────


def normalize_envelope(envelope: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a schema-v2 cognitive envelope with safe defaults.

    Accepts partial or malformed envelopes.  Unknown fields are preserved.

    Args:
        envelope: Input envelope dict (may be None or partial).

    Returns:
        Normalized envelope dict with canonical field names.
    """
    if not envelope or not isinstance(envelope, dict):
        return {}

    normalized: dict[str, Any] = dict(envelope)

    # Normalize intent
    intent = str(normalized.get("intent", "conversational")).lower()
    if intent not in INTENT_TYPES:
        intent = "conversational"
    normalized["intent"] = intent

    # Normalize coherence_score
    try:
        coherence = float(normalized.get("coherence_score", 1.0))
        normalized["coherence_score"] = max(0.0, min(1.0, coherence))
    except (TypeError, ValueError):
        normalized["coherence_score"] = 1.0

    # Normalize attention_budget
    try:
        budget = float(normalized.get("attention_budget", 1.0))
        normalized["attention_budget"] = max(0.0, min(1.0, budget))
    except (TypeError, ValueError):
        normalized["attention_budget"] = 1.0

    # Normalize governance mode
    gov = normalized.get("governance")
    if isinstance(gov, dict):
        mode = str(gov.get("governance_mode", "normal")).lower()
        if mode not in GOVERNANCE_MODES:
            mode = "normal"
        gov = dict(gov)
        gov["governance_mode"] = mode
        normalized["governance"] = gov

    return normalized


# ── Client ────────────────────────────────────────────────────────────────────


@dataclass
class SidecarClientConfig:
    """Connection configuration for a sidecar client."""
    unix_socket: str = ""
    tcp_host: str = ""
    tcp_port: int = 0
    http_base_url: str = "http://localhost:8000"
    token: str = ""
    timeout: int = DEFAULT_TIMEOUT
    output_mode: str = "pretty"  # pretty | json | raw


class SidecarClient:
    """Niblit sidecar IPC client.

    Supports UNIX socket, TCP, and HTTP transports.  Normalizes schema-v2
    envelopes on outbound requests and handles fragmented/malformed responses.

    Transport priority:
        1. UNIX socket (if unix_socket is set)
        2. TCP (if tcp_host + tcp_port > 0)
        3. HTTP (default)

    Args:
        config: :class:`SidecarClientConfig` instance.
    """

    def __init__(self, config: SidecarClientConfig | None = None) -> None:
        self._cfg = config or SidecarClientConfig()

    # ── High-level API ────────────────────────────────────────────────────────

    def health(self) -> SidecarResponse:
        """GET /health — liveness probe."""
        return self.get("/health")

    def runtime_status(self) -> SidecarResponse:
        """GET /v1/runtime/status — full runtime status."""
        return self.get("/v1/runtime/status")

    def diagnostics(self) -> SidecarResponse:
        """GET /v1/runtime/diagnostics — comprehensive diagnostics."""
        return self.get("/v1/runtime/diagnostics")

    def governance(self) -> SidecarResponse:
        """GET /v1/runtime/governance — governance stats."""
        return self.get("/v1/runtime/governance")

    def coherence(self) -> SidecarResponse:
        """GET /v1/runtime/coherence — temporal coherence."""
        return self.get("/v1/runtime/coherence")

    def federation_status(self) -> SidecarResponse:
        """GET /federation/status — federation manager status."""
        return self.get("/federation/status")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "local",
        temperature: float = 0.2,
        max_tokens: int = 256,
        envelope: dict[str, Any] | None = None,
    ) -> SidecarResponse:
        """POST /v1/chat/completions — run a chat inference request.

        Envelope fields are schema-v2 normalized before transmission.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if envelope:
            normalized = normalize_envelope(envelope)
            payload.update(normalized)
        return self.post("/v1/chat/completions", payload)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "local",
        temperature: float = 0.2,
        max_tokens: int = 256,
        envelope: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Stream a chat response token-by-token via SSE.

        Yields string tokens as they arrive.  Requires HTTP transport.
        Falls back to non-streaming chat if streaming is unavailable.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if envelope:
            payload.update(normalize_envelope(envelope))

        url = self._cfg.http_base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"

        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            return
                        try:
                            obj = json.loads(chunk)
                            delta = (
                                obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
        except Exception as exc:  # noqa: BLE001
            yield f"[stream error: {exc}]"

    # ── Low-level transport ───────────────────────────────────────────────────

    def get(self, path: str) -> SidecarResponse:
        return self._request("GET", path, None)

    def post(self, path: str, body: dict[str, Any]) -> SidecarResponse:
        return self._request("POST", path, body)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> SidecarResponse:
        t0 = time.perf_counter()
        cfg = self._cfg
        try:
            if cfg.unix_socket:
                resp = self._unix_request(method, path, body)
            elif cfg.tcp_host and cfg.tcp_port > 0:
                resp = self._tcp_request(method, path, body)
            else:
                resp = self._http_request(method, path, body)
        except Exception as exc:  # noqa: BLE001
            resp = SidecarResponse(
                ok=False, status_code=0, data={}, error=str(exc),
                url=self._url_for(path),
            )
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        return resp

    def _url_for(self, path: str) -> str:
        return self._cfg.http_base_url.rstrip("/") + path

    def _http_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> SidecarResponse:
        url = self._url_for(path)
        data = json.dumps(body).encode() if body is not None else None
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout) as resp:
                raw = resp.read().decode()
                return SidecarResponse(
                    ok=True, status_code=resp.status,
                    data=json.loads(raw) if raw else {}, url=url,
                )
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode()
                err_data = json.loads(raw) if raw else {}
            except Exception:  # noqa: BLE001
                err_data = {}
            return SidecarResponse(
                ok=False, status_code=exc.code, data=err_data, error=str(exc), url=url,
            )
        except Exception as exc:  # noqa: BLE001
            return SidecarResponse(ok=False, status_code=0, data={}, error=str(exc), url=url)

    def _unix_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> SidecarResponse:
        """Minimal HTTP/1.0 over UNIX domain socket."""
        url = f"unix:{self._cfg.unix_socket}{path}"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._cfg.timeout)
        try:
            sock.connect(self._cfg.unix_socket)
            body_bytes = json.dumps(body).encode() if body else b""
            headers = (
                f"Host: localhost\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
            )
            if self._cfg.token:
                headers += f"Authorization: Bearer {self._cfg.token}\r\n"
            raw = (f"{method} {path} HTTP/1.0\r\n" + headers + "\r\n").encode() + body_bytes
            sock.sendall(raw)

            response_data = b""
            while True:
                chunk = sock.recv(CHUNK_SIZE)
                if not chunk:
                    break
                response_data += chunk

            header_end = response_data.find(b"\r\n\r\n")
            if header_end == -1:
                return SidecarResponse(
                    ok=False, status_code=0, data={}, error="malformed http response", url=url,
                )
            header_part = response_data[:header_end].decode("utf-8", errors="replace")
            body_part = response_data[header_end + 4:]
            status_code = int(header_part.split("\r\n")[0].split(" ")[1])
            parsed_body = json.loads(body_part) if body_part.strip() else {}
            return SidecarResponse(
                ok=(200 <= status_code < 300),
                status_code=status_code,
                data=parsed_body,
                url=url,
            )
        except Exception as exc:  # noqa: BLE001
            return SidecarResponse(ok=False, status_code=0, data={}, error=str(exc), url=url)
        finally:
            sock.close()

    def _tcp_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> SidecarResponse:
        """JSON-over-TCP sidecar protocol."""
        url = f"tcp://{self._cfg.tcp_host}:{self._cfg.tcp_port}{path}"
        payload = {
            "method": method,
            "path": path,
            "body": body or {},
            "token": self._cfg.token,
            "protocol": SIDECAR_PROTOCOL_VERSION,
        }
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._cfg.timeout)
        try:
            sock.connect((self._cfg.tcp_host, self._cfg.tcp_port))
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            raw = b""
            while True:
                chunk = sock.recv(CHUNK_SIZE)
                if not chunk:
                    break
                raw += chunk
                # Stop on newline-terminated JSON (protocol guarantee)
                if raw.endswith(b"\n"):
                    break

            text = raw.decode("utf-8").strip()
            data = json.loads(text) if text else {}
            status_code = int(data.get("status_code", 200))
            return SidecarResponse(
                ok=(200 <= status_code < 300),
                status_code=status_code,
                data=data.get("data", data),
                error=str(data.get("error", "")),
                url=url,
            )
        except Exception as exc:  # noqa: BLE001
            return SidecarResponse(ok=False, status_code=0, data={}, error=str(exc), url=url)
        finally:
            sock.close()

    # ── Output rendering ──────────────────────────────────────────────────────

    def render(self, resp: SidecarResponse, mode: str | None = None) -> str:
        """Render a response according to the configured output mode.

        Args:
            resp: Response to render.
            mode: Override output mode (``pretty`` | ``json`` | ``raw``).

        Returns:
            Formatted string.
        """
        effective_mode = mode or self._cfg.output_mode
        if effective_mode == "json":
            return json.dumps(resp.data, separators=(",", ":"))
        if effective_mode == "raw":
            return str(resp.data)
        # pretty (default)
        return json.dumps(resp.data, indent=2, default=str)


# ── Convenience factory ───────────────────────────────────────────────────────


def from_env() -> SidecarClient:
    """Create a :class:`SidecarClient` from environment variables.

    Reads:
        ``NIBLIT_UNIX_SOCKET``      — UNIX socket path
        ``NIBLIT_TCP_ADMIN_HOST``   — TCP admin host
        ``NIBLIT_TCP_ADMIN_PORT``   — TCP admin port
        ``NIBLIT_CLOUD_URL``        — HTTP base URL
        ``NIBLIT_ADMIN_TOKEN``      — bearer token
    """
    import os

    return SidecarClient(
        SidecarClientConfig(
            unix_socket=os.getenv("NIBLIT_UNIX_SOCKET", ""),
            tcp_host=os.getenv("NIBLIT_TCP_ADMIN_HOST", ""),
            tcp_port=int(os.getenv("NIBLIT_TCP_ADMIN_PORT", "0")),
            http_base_url=os.getenv("NIBLIT_CLOUD_URL", "http://localhost:8000"),
            token=os.getenv("NIBLIT_ADMIN_TOKEN", ""),
        )
    )
