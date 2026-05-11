#!/usr/bin/env python3
"""tools/cloud_runtime_ctl.py — Niblit Cognitive Cloud Runtime Control CLI.

A single-file command-line tool for operating, diagnosing, and inspecting a
running Niblit Cognitive Cloud Runtime instance.

Usage
-----
    python tools/cloud_runtime_ctl.py [--url URL] <command> [args...]

Commands
--------
    health              — liveness probe
    status              — full runtime snapshot (all subsystems)
    coherence           — temporal coherence state
    governance          — constitutional governance stats
    attention           — attention economy metrics
    models              — model orchestration health
    reflection          — reflection engine telemetry
    trading             — trading cognition bridge state
    epoch               — current epoch
    cluster             — cluster / federation status
    diagnostics         — comprehensive diagnostics report
    chat <message>      — send a test chat message
    watch [interval]    — continuously poll runtime status (default 5s)
    ping                — quick connectivity check

Environment variables
---------------------
    NIBLIT_CLOUD_URL    — runtime base URL (default: http://localhost:8000)
    NIBLIT_ADMIN_TOKEN  — bearer token for authenticated endpoints
    NIBLIT_UNIX_SOCKET  — UNIX domain socket path (overrides HTTP)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running from repo root: python tools/cloud_runtime_ctl.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.lib.runtime_client import RuntimeClient  # noqa: E402


# ── Formatting helpers ────────────────────────────────────────────────────────


def _print_json(data: object, indent: int = 2) -> None:
    print(json.dumps(data, indent=indent, default=str))


def _print_section(title: str, data: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    _print_json(data)


def _status_line(name: str, ok: bool, detail: str = "") -> str:
    icon = "✅" if ok else "❌"
    line = f"  {icon}  {name}"
    if detail:
        line += f"  [{detail}]"
    return line


def _format_full_status(data: dict) -> None:
    """Pretty-print the full_status dict returned by RuntimeClient."""
    health = data.get("health", {})
    is_ok = health.get("status") == "ok"
    print(_status_line("Health", is_ok, health.get("status", "unknown")))

    runtime = data.get("runtime", {})
    print(_status_line("Runtime", bool(runtime), runtime.get("runtime", "")))

    coherence = data.get("coherence", {})
    c_ema = coherence.get("coherence_ema", "?")
    sync = coherence.get("sync_status", "?")
    print(_status_line("Temporal Sync", "error" not in coherence, f"coherence={c_ema} sync={sync}"))

    governance = data.get("governance", {})
    g_blocks = governance.get("block_count", 0)
    g_total = governance.get("validation_count", 0)
    print(_status_line("Governance", "error" not in governance, f"validated={g_total} blocked={g_blocks}"))

    attention = data.get("attention", {})
    pressure = attention.get("attention_pressure", "?")
    print(_status_line("Attention Economy", "error" not in attention, f"pressure={pressure}"))

    models = data.get("models", {})
    model_list = models.get("registered_models", [])
    print(_status_line("Model Orchestrator", "error" not in models, f"models={len(model_list)}"))

    reflection = data.get("reflection", {})
    quality = reflection.get("quality_ema", "?")
    print(_status_line("Reflection Engine", "error" not in reflection, f"quality_ema={quality}"))

    trading = data.get("trading", {})
    t_state = (trading.get("current_state") or {}).get("signal", "?")
    print(_status_line("Trading Bridge", "error" not in trading, f"signal={t_state}"))

    cluster = data.get("cluster", {})
    c_status = cluster.get("status", "?")
    print(_status_line("Cluster", "error" not in cluster, c_status))


# ── Command handlers ──────────────────────────────────────────────────────────


def cmd_ping(client: RuntimeClient) -> int:
    """Quick latency check."""
    t0 = time.perf_counter()
    resp = client.health()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if resp.ok:
        print(f"✅  PONG  {client.base_url}  ({elapsed_ms:.1f} ms)")
        return 0
    else:
        print(f"❌  FAIL  {client.base_url}  ({elapsed_ms:.1f} ms)  error={resp.error}")
        return 1


def cmd_health(client: RuntimeClient) -> int:
    resp = client.health()
    if resp.ok:
        print(f"✅  Runtime healthy  {client.base_url}")
        _print_json(resp.data)
        return 0
    print(f"❌  Runtime unhealthy  error={resp.error}")
    return 1


def cmd_status(client: RuntimeClient) -> int:
    print(f"\n🔍  Niblit Cognitive Cloud Runtime — Status")
    print(f"    URL: {client.base_url}\n")
    data = client.full_status()
    _format_full_status(data)
    print()
    return 0


def cmd_single(client: RuntimeClient, command: str) -> int:
    dispatch = {
        "coherence": client.coherence,
        "governance": client.governance,
        "attention": client.attention,
        "models": client.models,
        "reflection": client.reflection,
        "trading": client.trading,
        "epoch": client.epoch,
        "cluster": client.cluster_status,
        "diagnostics": client.diagnostics,
    }
    fn = dispatch.get(command)
    if fn is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 2
    resp = fn()
    if resp.ok:
        _print_json(resp.data)
        return 0
    print(f"❌  Error {resp.status_code}: {resp.error}", file=sys.stderr)
    _print_json(resp.data)
    return 1


def cmd_chat(client: RuntimeClient, message: str) -> int:
    print(f"→  Sending: {message!r}")
    resp = client.chat(messages=[{"role": "user", "content": message}])
    if resp.ok:
        choices = resp.data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            print(f"←  {content}")
        else:
            _print_json(resp.data)
        return 0
    print(f"❌  Error {resp.status_code}: {resp.error}", file=sys.stderr)
    return 1


def cmd_watch(client: RuntimeClient, interval: float = 5.0) -> int:
    """Continuously poll runtime status."""
    print(f"Watching {client.base_url}  (interval={interval}s)  Ctrl-C to stop\n")
    try:
        while True:
            t = time.strftime("%H:%M:%S")
            resp = client.runtime_status()
            if resp.ok:
                node = resp.data.get("node", {})
                temporal = resp.data.get("temporal", {})
                attention = resp.data.get("attention", {})
                uptime = node.get("uptime_secs", "?")
                reqs = node.get("request_count", "?")
                coh = temporal.get("coherence_ema", "?")
                pressure = attention.get("attention_pressure", "?")
                print(
                    f"[{t}] ✅  uptime={uptime}s  requests={reqs}  "
                    f"coherence={coh}  pressure={pressure}"
                )
            else:
                print(f"[{t}] ❌  {resp.error}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch stopped.")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloud_runtime_ctl",
        description="Niblit Cognitive Cloud Runtime control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=os.getenv("NIBLIT_CLOUD_URL", "http://localhost:8000"),
        help="Runtime base URL (default: http://localhost:8000 or NIBLIT_CLOUD_URL)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("NIBLIT_ADMIN_TOKEN", ""),
        help="Bearer token for authenticated endpoints",
    )
    parser.add_argument(
        "--socket",
        default=os.getenv("NIBLIT_UNIX_SOCKET", ""),
        help="UNIX domain socket path (overrides HTTP transport)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Request timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON (useful for piping)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping",        help="Quick connectivity latency check")
    sub.add_parser("health",      help="Liveness probe")
    sub.add_parser("status",      help="Full runtime snapshot")
    sub.add_parser("coherence",   help="Temporal coherence state")
    sub.add_parser("governance",  help="Constitutional governance stats")
    sub.add_parser("attention",   help="Attention economy metrics")
    sub.add_parser("models",      help="Model orchestration health")
    sub.add_parser("reflection",  help="Reflection engine telemetry")
    sub.add_parser("trading",     help="Trading cognition bridge state")
    sub.add_parser("epoch",       help="Current epoch and coherence")
    sub.add_parser("cluster",     help="Cluster / federation status")
    sub.add_parser("diagnostics", help="Comprehensive diagnostics report")

    chat_p = sub.add_parser("chat", help="Send a test chat message")
    chat_p.add_argument("message", nargs="?", default="Hello from cloud_runtime_ctl")

    watch_p = sub.add_parser("watch", help="Continuously poll runtime status")
    watch_p.add_argument(
        "interval",
        nargs="?",
        type=float,
        default=5.0,
        help="Poll interval in seconds (default: 5)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    client = RuntimeClient(
        base_url=args.url,
        timeout=args.timeout,
        admin_token=args.token,
        unix_socket=args.socket,
    )

    cmd = args.command

    if cmd == "ping":
        return cmd_ping(client)
    if cmd == "health":
        return cmd_health(client)
    if cmd == "status":
        if getattr(args, "json", False):
            data = client.full_status()
            _print_json(data)
            return 0
        return cmd_status(client)
    if cmd == "chat":
        return cmd_chat(client, args.message)
    if cmd == "watch":
        return cmd_watch(client, args.interval)
    if cmd in ("coherence", "governance", "attention", "models", "reflection",
               "trading", "epoch", "cluster", "diagnostics"):
        return cmd_single(client, cmd)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
