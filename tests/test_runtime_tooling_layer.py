"""tests/test_runtime_tooling_layer.py — Niblit Phase Ω.8 tooling layer tests.

Covers:
- Runtime profile loading (tools/lib/runtime_profiles.py)
- Sidecar client protocol normalization (tools/lib/sidecar_client.py)
- Local runtime validator (tools/install_local_runtime.py)
- Backward compatibility guarantees
- Profile completeness and governance-mode alignment
- Portability assertions
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Runtime profiles ───────────────────────────────────────────────────────────


class TestRuntimeProfiles:
    def test_list_profiles_returns_known_profiles(self):
        from tools.lib.runtime_profiles import KNOWN_PROFILES, list_profiles

        profiles = list_profiles()
        for p in KNOWN_PROFILES:
            assert p in profiles, f"Expected profile '{p}' in list_profiles()"

    def test_get_profile_env_cloud_server(self):
        from tools.lib.runtime_profiles import get_profile_env

        env = get_profile_env("cloud-server")
        assert env["NIBLIT_PROFILE"] == "cloud-server"
        assert env["NIBLIT_APP_NAME"] == "niblit-cloud-server"
        assert env["NIBLIT_PORT"] == "8000"

    def test_get_profile_env_termux_local(self):
        from tools.lib.runtime_profiles import get_profile_env

        env = get_profile_env("termux-local")
        assert env["NIBLIT_PROFILE"] == "termux-local"
        assert env["NIBLIT_RUNTIME_MODE"] == "cautious"
        n_ctx = int(env["NIBLIT_N_CTX"])
        assert n_ctx <= 4096, f"termux n_ctx too large: {n_ctx}"

    def test_get_profile_env_niblit(self):
        from tools.lib.runtime_profiles import get_profile_env

        env = get_profile_env("niblit")
        assert env["NIBLIT_PROFILE"] == "niblit"
        assert env["NIBLIT_GGUF_BACKEND"] == "http"

    def test_profile_not_found_raises(self):
        from tools.lib.runtime_profiles import ProfileNotFoundError, get_profile_env

        with pytest.raises(ProfileNotFoundError):
            get_profile_env("does-not-exist-profile")

    def test_load_profile_does_not_override_existing_env(self, monkeypatch):
        from tools.lib.runtime_profiles import load_profile

        monkeypatch.setenv("NIBLIT_PORT", "9999")
        load_profile("cloud-server", override=False)
        # Existing env var should not be overwritten
        assert os.environ["NIBLIT_PORT"] == "9999"

    def test_load_profile_override_updates_env(self, monkeypatch):
        from tools.lib.runtime_profiles import load_profile

        monkeypatch.setenv("NIBLIT_PORT", "9999")
        load_profile("cloud-server", override=True)
        # Should be overwritten with profile value
        assert os.environ["NIBLIT_PORT"] == "8000"

    def test_profile_summary_all_profiles(self):
        from tools.lib.runtime_profiles import KNOWN_PROFILES, profile_summary

        for profile in KNOWN_PROFILES:
            summary = profile_summary(profile)
            assert "error" not in summary, f"Profile '{profile}' has error"
            assert summary["runtime_mode"] in ("normal", "cautious", "survival", "lockdown"), \
                f"Profile '{profile}' has unexpected runtime_mode: {summary['runtime_mode']}"
            assert isinstance(summary["governance_strict"], bool)

    def test_cloud_server_profile_governance_strict(self):
        from tools.lib.runtime_profiles import profile_summary

        summary = profile_summary("cloud-server")
        assert summary["governance_strict"] is True

    def test_termux_profile_resource_class_minimal(self):
        from tools.lib.runtime_profiles import profile_summary

        summary = profile_summary("termux-local")
        assert summary["resource_class"] == "minimal"

    def test_degraded_and_disconnected_profiles(self):
        from tools.lib.runtime_profiles import profile_summary

        degraded = profile_summary("degraded-runtime")
        disconnected = profile_summary("disconnected-runtime")
        assert degraded["runtime_mode"] == "survival"
        assert disconnected["runtime_mode"] == "lockdown"

    def test_mode_normalization_aliases(self):
        from tools.lib.runtime_profiles import normalize_runtime_mode

        assert normalize_runtime_mode("minimal") == "cautious"
        assert normalize_runtime_mode("constrained") == "cautious"
        assert normalize_runtime_mode("lockdown") == "lockdown"

    def test_profile_resolver(self):
        from tools.lib.runtime_profiles import resolve_profile

        assert resolve_profile(topology="edge") == "edge-runtime"
        assert resolve_profile(topology="local") == "local-runtime"
        assert resolve_profile(degraded=True) == "degraded-runtime"
        assert resolve_profile(disconnected=True) == "disconnected-runtime"

    def test_profile_summary_includes_compatibility(self):
        from tools.lib.runtime_profiles import profile_summary

        summary = profile_summary("cloud-server")
        assert "compatibility" in summary
        assert summary["compatibility"]["schema_version"] == "2.x"

    def test_brace_expansion_in_profile_values(self):
        """Profile values like ${HOME}/models should expand."""
        from tools.lib.runtime_profiles import _expand_braces

        result = _expand_braces("${HOME}/models", {"HOME": "/test/home"})
        assert result == "/test/home/models"

    def test_brace_expansion_default_fallback(self):
        from tools.lib.runtime_profiles import _expand_braces

        result = _expand_braces("${MISSING:-/default/path}", {})
        assert result == "/default/path"

    def test_active_profile_default(self, monkeypatch):
        from tools.lib.runtime_profiles import active_profile

        monkeypatch.delenv("NIBLIT_PROFILE", raising=False)
        assert active_profile() == "cloud-server"

    def test_active_profile_from_env(self, monkeypatch):
        from tools.lib.runtime_profiles import active_profile

        monkeypatch.setenv("NIBLIT_PROFILE", "termux-local")
        assert active_profile() == "termux-local"

    def test_termux_inference_server_dry_run_resolves_termux_defaults(self, tmp_path):
        # Keep the GGUF stub larger than the 4-byte GGUF header so the launcher sees a non-empty model file.
        model_stub_size = 128
        repo_root = Path(__file__).parent.parent
        termux_home = tmp_path / "termux-home"
        model_dir = termux_home / "models"
        backend_bin = termux_home / "llama.cpp" / "build" / "bin" / "llama-server"
        tmp_runtime = tmp_path / "tmp"

        model_dir.mkdir(parents=True)
        backend_bin.parent.mkdir(parents=True)
        tmp_runtime.mkdir()

        model_path = model_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        model_path.write_bytes(b"GGUF" + b"\x00" * model_stub_size)
        backend_bin.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        backend_bin.chmod(0o755)

        env = os.environ.copy()
        env.update({
            "HOME": str(termux_home),
            "PREFIX": "/data/data/com.termux/files/usr",
            "TMPDIR": str(tmp_runtime),
        })
        env.pop("NIBLIT_MODEL_PATH", None)
        env.pop("NIBLIT_LLAMA_SERVER_BIN", None)

        result = subprocess.run(
            ["bash", str(repo_root / "tools/termux_inference_server.sh"), "--profile", "termux-local", "--dry-run"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert str(model_path) in result.stdout
        assert str(backend_bin) in result.stdout


# ── Sidecar client ─────────────────────────────────────────────────────────────


class TestSidecarClient:
    def test_protocol_constants(self):
        from tools.lib.sidecar_client import (
            CANONICAL_RUNTIME_MODES,
            DEFAULT_TIMEOUT,
            GOVERNANCE_MODES,
            INTENT_TYPES,
            SIDECAR_PROTOCOL_VERSION,
        )

        assert SIDECAR_PROTOCOL_VERSION.startswith("sidecar/")
        assert DEFAULT_TIMEOUT > 0
        assert "normal" in GOVERNANCE_MODES
        assert "lockdown" in CANONICAL_RUNTIME_MODES
        assert "trading" in INTENT_TYPES
        assert "conversational" in INTENT_TYPES

    def test_governance_modes_alignment(self):
        """Governance modes must match Ω.7 constants."""
        from tools.lib.sidecar_client import CANONICAL_RUNTIME_MODES

        expected = {"normal", "cautious", "survival", "lockdown"}
        assert expected == set(CANONICAL_RUNTIME_MODES), \
            f"Canonical mode drift: expected={expected} got={set(CANONICAL_RUNTIME_MODES)}"

    def test_normalize_envelope_identity(self):
        from tools.lib.sidecar_client import normalize_envelope

        envelope = {
            "intent": "trading",
            "coherence_score": 0.85,
            "attention_budget": 0.7,
            "governance": {"governance_mode": "normal"},
        }
        result = normalize_envelope(envelope)
        assert result["intent"] == "trading"
        assert result["coherence_score"] == 0.85
        assert result["attention_budget"] == 0.7
        assert result["governance"]["governance_mode"] == "normal"

    def test_normalize_envelope_clamps_coherence(self):
        from tools.lib.sidecar_client import normalize_envelope

        result = normalize_envelope({"coherence_score": 1.5})
        assert result["coherence_score"] == 1.0

        result2 = normalize_envelope({"coherence_score": -0.1})
        assert result2["coherence_score"] == 0.0

    def test_normalize_envelope_unknown_intent_defaults(self):
        from tools.lib.sidecar_client import normalize_envelope

        result = normalize_envelope({"intent": "something_unknown"})
        assert result["intent"] == "conversational"

    def test_normalize_envelope_unknown_governance_mode_defaults(self):
        from tools.lib.sidecar_client import normalize_envelope

        result = normalize_envelope({"governance": {"governance_mode": "invalid_mode"}})
        assert result["governance"]["governance_mode"] == "normal"

    def test_normalize_envelope_none_returns_empty(self):
        from tools.lib.sidecar_client import normalize_envelope

        assert normalize_envelope(None) == {}
        assert normalize_envelope({}) == {}

    def test_normalize_envelope_preserves_unknown_fields(self):
        from tools.lib.sidecar_client import normalize_envelope

        result = normalize_envelope({"custom_field": "preserved", "intent": "trading"})
        assert result["custom_field"] == "preserved"

    def test_sidecar_client_instantiation(self):
        from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig

        client = SidecarClient(SidecarClientConfig(http_base_url="http://localhost:9999"))
        assert client._cfg.http_base_url == "http://localhost:9999"

    def test_sidecar_compatibility_checker(self):
        from tools.lib.sidecar_client import check_compatibility

        ok = check_compatibility({"compatibility": {"schema_version": "2.x"}})
        assert ok["compatible"]

        bad = check_compatibility({"compatibility": {"schema_version": "1.x"}})
        assert not bad["compatible"]
        assert "schema_version" in bad["mismatches"]

    def test_sidecar_response_repr(self):
        from tools.lib.sidecar_client import SidecarResponse

        r_ok = SidecarResponse(ok=True, status_code=200, data={}, latency_ms=5.0)
        assert "ok" in repr(r_ok)

        r_err = SidecarResponse(ok=False, status_code=0, data={}, error="conn refused")
        assert "conn refused" in repr(r_err)

    def test_render_json_mode(self):
        from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig, SidecarResponse

        client = SidecarClient(SidecarClientConfig(output_mode="json"))
        resp = SidecarResponse(ok=True, status_code=200, data={"key": "val"})
        rendered = client.render(resp, mode="json")
        assert json.loads(rendered) == {"key": "val"}

    def test_render_pretty_mode(self):
        from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig, SidecarResponse

        client = SidecarClient(SidecarClientConfig())
        resp = SidecarResponse(ok=True, status_code=200, data={"a": 1})
        rendered = client.render(resp, mode="pretty")
        assert '"a"' in rendered

    def test_from_env_factory_defaults(self, monkeypatch):
        from tools.lib.sidecar_client import SidecarClient, from_env

        monkeypatch.delenv("NIBLIT_UNIX_SOCKET", raising=False)
        monkeypatch.delenv("NIBLIT_TCP_ADMIN_HOST", raising=False)
        monkeypatch.delenv("NIBLIT_TCP_ADMIN_PORT", raising=False)
        monkeypatch.setenv("NIBLIT_CLOUD_URL", "http://test-host:7777")
        monkeypatch.delenv("NIBLIT_ADMIN_TOKEN", raising=False)

        client = from_env()
        assert isinstance(client, SidecarClient)
        assert client._cfg.http_base_url == "http://test-host:7777"
        assert client._cfg.unix_socket == ""

    def test_from_env_factory_with_socket(self, monkeypatch):
        from tools.lib.sidecar_client import from_env

        monkeypatch.setenv("NIBLIT_UNIX_SOCKET", "/tmp/test.sock")
        client = from_env()
        assert client._cfg.unix_socket == "/tmp/test.sock"

    def test_http_transport_handles_connection_error(self):
        """HTTP requests to unreachable host return error SidecarResponse."""
        from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig

        client = SidecarClient(
            SidecarClientConfig(http_base_url="http://127.0.0.1:1", timeout=1)
        )
        resp = client.health()
        assert not resp.ok
        assert resp.status_code == 0
        assert resp.error != ""

    def test_tcp_transport_handles_connection_error(self):
        """TCP requests to unreachable port return error SidecarResponse."""
        from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig

        client = SidecarClient(
            SidecarClientConfig(tcp_host="127.0.0.1", tcp_port=1, timeout=1)
        )
        resp = client.get("/health")
        assert not resp.ok
        assert resp.status_code == 0

    def test_unix_socket_handles_missing_socket(self):
        """UNIX socket requests to non-existent socket return error SidecarResponse."""
        from tools.lib.sidecar_client import SidecarClient, SidecarClientConfig

        client = SidecarClient(
            SidecarClientConfig(unix_socket="/tmp/niblit-nonexistent-test.sock", timeout=1)
        )
        resp = client.get("/health")
        assert not resp.ok


# ── Local runtime validator ────────────────────────────────────────────────────


class TestRuntimeValidator:
    def test_detect_platform_returns_expected_keys(self):
        from tools.install_local_runtime import detect_platform

        pf = detect_platform()
        for key in ("os", "arch", "python_version", "is_termux", "is_container",
                    "is_linux", "is_arm", "is_x86", "cpu_count"):
            assert key in pf, f"detect_platform() missing key: {key}"

    def test_detect_platform_not_termux_in_ci(self):
        from tools.install_local_runtime import detect_platform

        pf = detect_platform()
        # CI is not a Termux environment
        assert not pf["is_termux"], "CI should not be detected as Termux"

    def test_validate_env_returns_dict(self):
        from tools.install_local_runtime import validate_env

        result = validate_env()
        assert isinstance(result, dict)
        assert "ok" in result
        assert "missing_recommended" in result

    def test_validate_backend_no_binary(self):
        """In CI, llama-server is not installed — should return not-ok."""
        from tools.install_local_runtime import validate_backend

        result = validate_backend()
        # May be ok (if accidentally installed in CI) or not ok
        assert "ok" in result
        if not result["ok"]:
            assert "error" in result

    def test_validate_model_missing_file(self):
        from tools.install_local_runtime import validate_model

        result = validate_model("/tmp/nonexistent-model-file.gguf")
        assert not result["ok"]
        assert "error" in result

    def test_validate_model_existing_non_gguf(self, tmp_path):
        from tools.install_local_runtime import validate_model

        fake = tmp_path / "test.gguf"
        fake.write_bytes(b"NOTGGUF" + b"\x00" * 1024)

        result = validate_model(str(fake))
        assert result["ok"]  # File exists → ok
        assert not result["is_gguf"]  # But not GGUF magic
        assert len(result["warnings"]) > 0

    def test_validate_model_gguf_magic(self, tmp_path):
        from tools.install_local_runtime import validate_model

        fake = tmp_path / "model.gguf"
        # GGUF magic + fake content
        fake.write_bytes(b"GGUF" + b"\x00" * (200 * 1024))

        result = validate_model(str(fake))
        assert result["ok"]
        assert result["is_gguf"]

    def test_detect_model_family_qwen(self):
        from tools.install_local_runtime import detect_model_family

        assert detect_model_family("/models/qwen2.5-7b-instruct-q4_k_m.gguf") == "qwen"

    def test_detect_model_family_llama(self):
        from tools.install_local_runtime import detect_model_family

        assert detect_model_family("/models/llama-3-8b.gguf") == "llama"

    def test_detect_model_family_unknown(self):
        from tools.install_local_runtime import detect_model_family

        assert detect_model_family("/models/custom-model.gguf") == "unknown"

    def test_install_hints_not_empty(self):
        from tools.install_local_runtime import detect_platform, install_hints

        pf = detect_platform()
        hints = install_hints(pf)
        assert isinstance(hints, list)
        assert len(hints) > 0

    def test_find_models_in_dir_nonexistent(self, tmp_path):
        from tools.install_local_runtime import find_models_in_dir

        result = find_models_in_dir(str(tmp_path / "nonexistent"))
        assert isinstance(result, list)

    def test_find_models_in_dir_with_gguf(self, tmp_path):
        from tools.install_local_runtime import find_models_in_dir

        (tmp_path / "model.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
        result = find_models_in_dir(str(tmp_path))
        assert any("model.gguf" in r for r in result)

    def test_main_info_exits_zero(self):
        from tools.install_local_runtime import main

        rc = main(["--info"])
        assert rc == 0

    def test_main_check_backend_exits_cleanly(self):
        from tools.install_local_runtime import main

        # Should not raise — either 0 (found) or 1 (not found)
        rc = main(["--check-backend"])
        assert rc in (0, 1)

    def test_main_check_model_missing_exits_nonzero(self):
        from tools.install_local_runtime import main

        rc = main(["--check-model", "/tmp/totally-nonexistent.gguf"])
        assert rc == 1


# ── Backward compatibility ─────────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Verify that Phase Ω.8 tooling additions don't break existing contracts."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        from app.main import ModelEngineResult, ModelManager, create_app

        class FakeManager(ModelManager):
            def __init__(self):
                super().__init__(
                    model_map={"demo": "/tmp/demo.gguf"},
                    default_model="demo",
                )

            def chat(self, model_id, messages, temperature, max_tokens):
                return ModelEngineResult(
                    text="ok",
                    finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

        app = create_app(FakeManager())
        return TestClient(app), FakeManager()

    def test_chat_completions_still_works(self):
        client, _ = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert "choices" in resp.json()

    def test_health_endpoint_still_works(self):
        client, _ = self._make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_federation_status_endpoint_works(self):
        client, _ = self._make_client()
        resp = client.get("/federation/status")
        assert resp.status_code == 200

    def test_diagnostics_endpoint_works(self):
        client, _ = self._make_client()
        resp = client.get("/v1/runtime/diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        assert "runtime_health" in data

    def test_runtime_topology_endpoint_works(self):
        client, _ = self._make_client()
        resp = client.get("/v1/runtime/topology")
        assert resp.status_code == 200
        data = resp.json()
        assert "compatibility" in data
        assert "runtime_mode" in data

    def test_runtime_client_and_sidecar_client_are_independent(self):
        """RuntimeClient and SidecarClient must be importable independently."""
        from tools.lib import runtime_client, sidecar_client

        assert hasattr(runtime_client, "RuntimeClient")
        assert hasattr(sidecar_client, "SidecarClient")

    def test_niblit_ctl_wrapper_importable(self):
        from tools import niblit_ctl

        assert callable(niblit_ctl.main)

    def test_runtime_profiles_do_not_modify_env_on_import(self):
        """Importing runtime_profiles must not side-effect os.environ."""
        import os
        before = dict(os.environ)
        import tools.lib.runtime_profiles  # noqa: F401
        after = dict(os.environ)
        assert before == after, "runtime_profiles import modified os.environ"


# ── Tunnel URL parsing ─────────────────────────────────────────────────────────


class TestTunnelParsing:
    """Validate tunnel URL detection patterns."""

    def test_cloudflared_url_pattern(self):
        import re
        pattern = r"https://[a-z0-9.-]*\.trycloudflare\.com"
        sample_log = "2026-01-01T00:00:00Z INF | https://example-tunnel.trycloudflare.com"
        match = re.search(pattern, sample_log)
        assert match is not None
        assert match.group(0).startswith("https://")

    def test_cloudflared_no_match_for_local(self):
        import re
        pattern = r"https://[a-z0-9.-]*\.trycloudflare\.com"
        assert not re.search(pattern, "http://localhost:8000")

    def test_public_url_override_accepted(self):
        """NIBLIT_TUNNEL_PUBLIC_URL env var should be readable."""
        url = "https://my-tunnel.example.com"
        os.environ["NIBLIT_TUNNEL_PUBLIC_URL"] = url
        assert os.environ.get("NIBLIT_TUNNEL_PUBLIC_URL") == url
        del os.environ["NIBLIT_TUNNEL_PUBLIC_URL"]
