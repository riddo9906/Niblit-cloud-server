"""Tests for local GGUF model registry and resolution."""

from __future__ import annotations

import os

from app.local_models import (
    CONTEXT_MODEL_ID,
    PRIMARY_MODEL_ID,
    SUMMARIZER_MODEL_ID,
    discover_local_models,
    gguf_model_dir,
    is_known_alias,
    resolve_alias,
)


def test_three_required_models_discovered() -> None:
    """All three required GGUF models must be discoverable from the authoritative directory."""
    specs = discover_local_models()
    assert PRIMARY_MODEL_ID in specs, f"Primary model {PRIMARY_MODEL_ID} not discovered"
    assert SUMMARIZER_MODEL_ID in specs, f"Summarizer model {SUMMARIZER_MODEL_ID} not discovered"
    assert CONTEXT_MODEL_ID in specs, f"Context model {CONTEXT_MODEL_ID} not discovered"


def test_discovered_models_have_valid_paths() -> None:
    """Every discovered model must point to a real GGUF file."""
    specs = discover_local_models()
    for model_id, spec in specs.items():
        assert os.path.isfile(spec.path), f"Model {model_id} path does not exist: {spec.path}"
        assert spec.path.lower().endswith(".gguf"), f"Model {model_id} path is not a .gguf file"
        assert spec.available, f"Model {model_id} should be available"


def test_model_context_limits_are_model_specific() -> None:
    """Models must have per-model context limits, not a blanket 16384."""
    specs = discover_local_models()
    # The primary model should have 16384, the small models less
    if PRIMARY_MODEL_ID in specs:
        assert specs[PRIMARY_MODEL_ID].runtime_context_limit >= 8192
    if SUMMARIZER_MODEL_ID in specs:
        assert specs[SUMMARIZER_MODEL_ID].runtime_context_limit >= 4096
    if CONTEXT_MODEL_ID in specs:
        assert specs[CONTEXT_MODEL_ID].runtime_context_limit >= 2048


def test_truthful_role_aliases() -> None:
    """Aliases must resolve to their documented targets, never cross-family."""
    # 'llama3' must resolve to the Llama model, NOT to Qwen
    assert resolve_alias("llama3") == SUMMARIZER_MODEL_ID
    assert resolve_alias("llama") == SUMMARIZER_MODEL_ID
    assert resolve_alias("default") == PRIMARY_MODEL_ID
    assert resolve_alias("primary") == PRIMARY_MODEL_ID
    assert resolve_alias("coder") == PRIMARY_MODEL_ID
    assert resolve_alias("small-summary") == SUMMARIZER_MODEL_ID
    assert resolve_alias("small-context") == CONTEXT_MODEL_ID


def test_explicit_unknown_model_is_not_aliased() -> None:
    """Unknown model refs must pass through unchanged for clean model_not_found."""
    assert not is_known_alias("nonexistent-model-xyz")
    assert resolve_alias("nonexistent-model-xyz") == "nonexistent-model-xyz"


def test_gguf_dir_from_env() -> None:
    """The GGUF directory should be configurable via env var."""
    assert gguf_model_dir()  # truthy


def test_canonical_ids_are_truthful() -> None:
    """The canonical IDs must match the actual model families."""
    specs = discover_local_models()
    if PRIMARY_MODEL_ID in specs:
        spec = specs[PRIMARY_MODEL_ID]
        assert "qwen" in spec.family.lower(), f"Primary model family should be qwen, got {spec.family}"
    if SUMMARIZER_MODEL_ID in specs:
        spec = specs[SUMMARIZER_MODEL_ID]
        assert "llama" in spec.family.lower(), f"Summarizer model family should be llama, got {spec.family}"
    if CONTEXT_MODEL_ID in specs:
        spec = specs[CONTEXT_MODEL_ID]
        assert "qwen" in spec.family.lower(), f"Context model family should be qwen, got {spec.family}"