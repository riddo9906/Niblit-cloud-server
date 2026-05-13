#!/usr/bin/env python3
"""Thin CLI wrapper over tools.lib.sidecar_client for runtime coordination."""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig, from_env  # noqa: E402


def _print(resp, output_mode: str, client: SidecarClient) -> int:
    print(client.render(resp, mode=output_mode))
    return 0 if resp.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="niblit_ctl", description="Niblit sidecar/runtime control CLI")
    parser.add_argument("--url", default=os.getenv("NIBLIT_CLOUD_URL", "http://localhost:8000"))
    parser.add_argument("--socket", default=os.getenv("NIBLIT_UNIX_SOCKET", ""))
    parser.add_argument("--tcp-host", default=os.getenv("NIBLIT_TCP_ADMIN_HOST", ""))
    parser.add_argument("--tcp-port", type=int, default=int(os.getenv("NIBLIT_TCP_ADMIN_PORT", "0")))
    parser.add_argument("--token", default=os.getenv("NIBLIT_ADMIN_TOKEN", ""))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("NIBLIT_TIMEOUT", "15")))
    parser.add_argument("--output", choices=["pretty", "json", "raw"], default="pretty")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    sub.add_parser("status")
    sub.add_parser("diagnostics")
    sub.add_parser("governance")
    sub.add_parser("coherence")
    sub.add_parser("federation")
    sub.add_parser("topology")
    sub.add_parser("compatibility")

    chat = sub.add_parser("chat")
    chat.add_argument("message", nargs="?", default="hello from niblit_ctl")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    client = SidecarClient(
        SidecarClientConfig(
            unix_socket=args.socket,
            tcp_host=args.tcp_host,
            tcp_port=args.tcp_port,
            http_base_url=args.url,
            token=args.token,
            timeout=args.timeout,
            output_mode=args.output,
        )
    )

    if args.command == "health":
        return _print(client.health(), args.output, client)
    if args.command == "status":
        return _print(client.runtime_status(), args.output, client)
    if args.command == "diagnostics":
        return _print(client.diagnostics(), args.output, client)
    if args.command == "governance":
        return _print(client.governance(), args.output, client)
    if args.command == "coherence":
        return _print(client.coherence(), args.output, client)
    if args.command == "federation":
        return _print(client.federation_status(), args.output, client)
    if args.command == "topology":
        return _print(client.topology(), args.output, client)
    if args.command == "compatibility":
        return _print(client.compatibility(), args.output, client)
    if args.command == "chat":
        return _print(
            client.chat(messages=[{"role": "user", "content": args.message}]),
            args.output,
            client,
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
