"""Centralized base-URL normalization for Niblit Cloud.

Invariant:
    server root  = http://127.0.0.1:8000          (no ``/v1`` suffix)
    OpenAI base  = http://127.0.0.1:8000/v1       (exactly one ``/v1``)

Clients configured with an OpenAI base may pass a URL ending in ``/v1``;
internal raw HTTP helpers always use the server root. These helpers are the
ONLY sanctioned place where a ``/v1`` suffix is added or removed — never
scatter ``rstrip("/v1")`` hacks at call sites.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def normalize_server_root(base_url: str) -> str:
    """Return a server root URL with no path (in particular no ``/v1``).

    Examples:
        http://127.0.0.1:8000        -> http://127.0.0.1:8000
        http://127.0.0.1:8000/       -> http://127.0.0.1:8000
        http://127.0.0.1:8000/v1     -> http://127.0.0.1:8000
        http://127.0.0.1:8000/v1/    -> http://127.0.0.1:8000
        http://localhost:8000/v1/v1/ -> http://localhost:8000   (defensive)
    """
    value = (base_url or "").strip().rstrip("/")
    if not value:
        return value
    parts = urlsplit(value)
    path = parts.path
    while True:
        stripped = path.rstrip("/")
        if len(stripped) >= 3 and stripped[-3:].lower() == "/v1":
            path = stripped[:-3]
        else:
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", "")).rstrip("/") or urlunsplit(
        (parts.scheme, parts.netloc, "", "", "")
    )


def normalize_openai_base_url(base_url: str) -> str:
    """Return an OpenAI base URL with exactly one trailing ``/v1``.

    Examples:
        http://127.0.0.1:8000     -> http://127.0.0.1:8000/v1
        http://127.0.0.1:8000/v1  -> http://127.0.0.1:8000/v1
        http://127.0.0.1:8000/v1/ -> http://127.0.0.1:8000/v1
    """
    root = normalize_server_root(base_url)
    if not root:
        return root
    return f"{root}/v1"


def join_server_root(server_root: str, api_path: str) -> str:
    """Join a server root with an API path such as ``/v1/models``."""
    root = normalize_server_root(server_root)
    path = api_path if api_path.startswith("/") else f"/{api_path}"
    return f"{root}{path}"


def join_openai_base(openai_base: str, endpoint: str) -> str:
    """Join an OpenAI base (…/v1) with an endpoint such as ``/models``.

    Accepts endpoints given either with or without the leading ``/v1`` so that
    a misconfigured caller can never produce ``/v1/v1/...``.
    """
    base = normalize_openai_base_url(openai_base)
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if path == "/v1" or path.startswith("/v1/"):
        path = path[len("/v1"):]
    return f"{base}{path}"
