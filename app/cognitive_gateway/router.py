"""Rule-based model routing for the Cognitive Gateway."""

from __future__ import annotations

from dataclasses import dataclass

from app.cognitive_gateway.config import CognitiveGatewayConfig

_LOCAL_ALIASES: frozenset[str] = frozenset({"local", "llama", "default", ""})


@dataclass(frozen=True)
class RoutingDecision:
    model_id: str | None
    source: str
    fallback_used: bool = False


def _pick_available(candidate: str | None, available: set[str]) -> str | None:
    if candidate and candidate in available:
        return candidate
    return None


def route_model(
    *,
    config: CognitiveGatewayConfig,
    request_type: str,
    available_models: list[str],
    requested_model: str | None,
    default_model: str | None,
) -> RoutingDecision:
    """Select model ID based on routing strategy and classification."""
    available = set(available_models)
    explicit = (requested_model or "").strip()
    explicit_is_alias = explicit.lower() in _LOCAL_ALIASES if explicit else False
    explicit_valid = explicit and not explicit_is_alias and explicit in available

    if config.routing_strategy == "passthrough":
        if explicit_valid:
            return RoutingDecision(model_id=explicit, source="passthrough")
        model = default_model
        return RoutingDecision(model_id=model, source="passthrough")

    if config.routing_strategy == "static":
        static = config.routing_rules.get("static") or config.fallback_model or default_model
        chosen = _pick_available(static, available)
        return RoutingDecision(model_id=chosen, source="static")

    # dynamic strategy
    if config.respect_explicit_model:
        if explicit_is_alias:
            return RoutingDecision(model_id=default_model, source="default", fallback_used=True)
        if explicit:
            if explicit_valid:
                return RoutingDecision(model_id=explicit, source="explicit")
            # Explicit unknown model — preserve None so caller can raise 404.
            return RoutingDecision(model_id=None, source="explicit_unknown")

    rule_model = config.routing_rules.get(request_type)
    chosen = _pick_available(rule_model, available)
    if chosen:
        return RoutingDecision(model_id=chosen, source=f"rule:{request_type}")

    fallback = _pick_available(config.fallback_model, available)
    if fallback:
        return RoutingDecision(model_id=fallback, source="fallback", fallback_used=True)

    return RoutingDecision(model_id=default_model, source="default", fallback_used=True)
