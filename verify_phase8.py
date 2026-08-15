"""Phase 8 verification script."""
from app.runtime import CloudRuntime

r = CloudRuntime()
r.run()

ready = any(s.stage.value == "ready" and s.status == "completed" for s in r.stages)
degraded = [s.stage.value for s in r.stages if s.status == "degraded"]
failed = [s.stage.value for s in r.stages if s.status == "failed"]
services = list(r._state.keys())

key_services = [
    "evaluation_engine", "policy_optimizer", "autonomous_learning_engine",
    "lean_algo_manager", "reflection_engine", "strategy_evolution",
    "execution_feedback", "trading_pipeline", "freqtrade_adapter",
    "risk_engine", "market_intelligence", "embedding_service", "vector_db",
]

print("READY" if ready else "FAILED")
print("Degraded:", degraded)
print("Failed:", failed)
print("Services:", len(services))
print("Key services present:", all(s in services for s in key_services))
</arg_value>
</write_to_file>