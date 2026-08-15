#!/usr/bin/env python3
"""Niblit Cloud Server — canonical entrypoint.

Running:

    python main.py

executes the full layered CloudRuntime boot sequence and then starts the
Uvicorn HTTP server.  No shell scripts are required.
"""

from app.main import _run_cli

if __name__ == "__main__":
    _run_cli()