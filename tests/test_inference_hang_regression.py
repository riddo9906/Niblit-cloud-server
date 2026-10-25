"""Regression test for the Cloud inference hang.

Root cause: ModelManager.chat() acquired self._lock (a non-reentrant
threading.Lock) and then called self.model_context_limit() inside the
critical section; model_context_limit() re-acquires the same lock, so
every cold-engine request self-deadlocked forever (health/models stayed
fine, native llama.cpp stayed healthy).
"""
import threading

from app import main as app_main


class _DummyEngine:
    instances: list["_DummyEngine"] = []

    def __init__(self, model_path: str, n_ctx: int, n_threads: int, runtime_options=None):
        self.model_path = model_path
        self.n_ctx = n_ctx
        _DummyEngine.instances.append(self)

    def chat(self, messages, temperature, max_tokens, **kwargs):
        return app_main.ModelEngineResult(
            text="OK", finish_reason="stop", usage={"completion_tokens": 2}
        )


def test_cold_engine_chat_does_not_self_deadlock(monkeypatch):
    """A cold-model request must complete instead of blocking on its own lock."""
    monkeypatch.setattr(app_main, "GGUFEngine", _DummyEngine)
    _DummyEngine.instances.clear()
    mgr = app_main.ModelManager({"test-model": "fake.gguf"}, "test-model")

    box: dict = {}

    def run():
        box["result"] = mgr.chat(
            model_id="test-model",
            messages=[{"role": "user", "content": "Reply with exactly OK"}],
            temperature=0.0,
            max_tokens=16,
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=15)

    assert not worker.is_alive(), (
        "manager.chat() deadlocked: model_context_limit() re-acquired the "
        "non-reentrant manager lock inside the engine-creation critical section"
    )
    assert box["result"].text == "OK"
    # Engine must have been created with the model-specific context limit.
    assert _DummyEngine.instances[-1].n_ctx == mgr.model_context_limit("test-model")


def test_warm_engine_chat_still_works(monkeypatch):
    """Warm-engine path (no creation) remains functional."""
    monkeypatch.setattr(app_main, "GGUFEngine", _DummyEngine)
    _DummyEngine.instances.clear()
    mgr = app_main.ModelManager({"test-model": "fake.gguf"}, "test-model")
    first = mgr.chat(model_id="test-model", messages=[{"role": "user", "content": "hi"}],
                     temperature=0.0, max_tokens=8)
    second = mgr.chat(model_id="test-model", messages=[{"role": "user", "content": "hi"}],
                      temperature=0.0, max_tokens=8)
    assert first.text == second.text == "OK"
    assert len(_DummyEngine.instances) == 1  # no duplicate loads

