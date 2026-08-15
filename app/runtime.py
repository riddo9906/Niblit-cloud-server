"""Niblit Cloud Runtime — Central lifecycle manager.

Coordinates module discovery, initialization, health verification, and
graceful shutdown across all cognitive subsystems.

All subsystems consume the same CloudConfig instance passed at construction.
No legacy env-var lookups. No duplicate runtime creation.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BootStage(str, Enum):
    CONFIGURATION = "configuration"
    STORAGE = "storage"
    MODEL_REGISTRY = "model_registry"
    INFERENCE = "inference"
    EMBEDDINGS = "embeddings"
    VECTOR_DATABASE = "vector_database"
    API_SERVICES = "api_services"
    WEBSOCKET = "websocket"
    SCHEDULER = "scheduler"
    BACKGROUND_SERVICES = "background_services"
    PLUGINS = "plugins"
    MARKET_DATA = "market_data"
    MARKET_STATE = "market_state"
    MEMORY = "memory"
    LEAN_ALGORITHMS = "lean_algorithms"
    LLM_VALIDATION = "llm_validation"
    RISK_ENGINE = "risk_engine"
    FREQTRADE = "freqtrade"
    EXECUTION_MONITOR = "execution_monitor"
    HEALTH = "health"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class StageResult:
    stage: BootStage
    status: str  # completed|degraded|failed
    message: str = ""
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class CloudRuntime:
    """Coordinates startup of every Cloud Server subsystem in a reproducible order.

    Accepts an optional CloudConfig at construction.  If omitted, loads from
    the global singleton.  Every subsystem receives the same config instance.
    """

    def __init__(self, config: Any | None = None) -> None:
        if config is None:
            from app.config import get_config
            config = get_config()
        self._config = config
        self._stages: list[StageResult] = []
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._shutdown_callbacks: list[Any] = []
        logger.info("CloudRuntime instantiated (config id=%d)", id(config))

    # ── context manager ─────────────────────────────────────────────────────────

    def __enter__(self) -> CloudRuntime:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: A002
        self.shutdown()

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self) -> bool:
        """Execute the full boot sequence. Returns True if READY."""
        logger.info("Cloud Server Boot")
        self._run_stage(BootStage.CONFIGURATION, self._boot_configuration)
        self._run_stage(BootStage.STORAGE, self._boot_storage)
        self._run_stage(BootStage.MODEL_REGISTRY, self._boot_model_registry)
        self._run_stage(BootStage.INFERENCE, self._boot_inference)
        self._run_stage(BootStage.EMBEDDINGS, self._boot_embeddings)
        self._run_stage(BootStage.VECTOR_DATABASE, self._boot_vector_database)
        self._run_stage(BootStage.API_SERVICES, self._boot_api_services)
        self._run_stage(BootStage.WEBSOCKET, self._boot_websocket)
        self._run_stage(BootStage.SCHEDULER, self._boot_scheduler)
        self._run_stage(BootStage.BACKGROUND_SERVICES, self._boot_background_services)
        self._run_stage(BootStage.PLUGINS, self._boot_plugins)
        self._run_stage(BootStage.MARKET_DATA, self._boot_market_data)
        self._run_stage(BootStage.MARKET_STATE, self._boot_market_state)
        self._run_stage(BootStage.MEMORY, self._boot_memory)
        self._run_stage(BootStage.LEAN_ALGORITHMS, self._boot_lean_algorithms)
        self._run_stage(BootStage.LLM_VALIDATION, self._boot_llm_validation)
        self._run_stage(BootStage.RISK_ENGINE, self._boot_risk_engine)
        self._run_stage(BootStage.FREQTRADE, self._boot_freqtrade)
        self._run_stage(BootStage.EXECUTION_MONITOR, self._boot_execution_monitor)
        self._run_stage(BootStage.HEALTH, self._boot_health)
        ready = self._finalize()
        return ready

    def shutdown(self) -> None:
        """Best-effort graceful shutdown."""
        logger.info("CloudRuntime shutdown: stopping services")
        for shutdown_fn in reversed(self._shutdown_callbacks):
            try:
                shutdown_fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shutdown callback failed: %s", exc)

    @property
    def stages(self) -> list[StageResult]:
        with self._lock:
            return list(self._stages)

    def get_service(self, name: str) -> Any:
        return self._state.get(name)

    def register(self, name: str, value: Any) -> None:
        with self._lock:
            self._state[name] = value

    def register_shutdown(self, fn) -> None:
        self._shutdown_callbacks.append(fn)

    @property
    def config(self) -> Any:
        return self._config

    # ── stage runner ───────────────────────────────────────────────────────────

    def _run_stage(self, stage: BootStage, fn) -> None:
        logger.info("[%s] starting", stage.value)
        started = time.perf_counter()
        try:
            detail = fn()
            duration_ms = (time.perf_counter() - started) * 1000
            result = StageResult(
                stage=stage,
                status="completed",
                message=str(detail) if detail is not None else "",
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning("CloudRuntime stage %s failed: %s", stage.value, exc)
            result = StageResult(
                stage=stage,
                status="failed",
                message=str(exc),
                duration_ms=duration_ms,
            )
        self._stages.append(result)
        level = logging.ERROR if result.status == "failed" else logging.INFO
        logger.log(
            level,
            "[%s] %s (%.1f ms): %s",
            stage.value,
            result.status,
            result.duration_ms,
            result.message,
        )

    # ── boot functions ─────────────────────────────────────────────────────────

    def _boot_configuration(self) -> str:
        """Layer 1: Configuration — uses CloudConfig directly."""
        cfg = self._config
        models = cfg.model_map
        default_model = cfg.default_model
        self.register("model_map", models)
        self.register("default_model", default_model)
        self.register("n_ctx", cfg.n_ctx)
        self.register("n_threads", cfg.n_threads)
        self.register("n_batch", cfg.n_batch)
        self.register("n_ubatch", cfg.n_ubatch)
        return f"models={len(models)} default={default_model or '(none)'}"

    def _boot_storage(self) -> str:
        """Layer 2: Storage — filesystem-backed for now."""
        root = self._config.storage_root or os.path.join(os.getcwd(), ".niblit", "storage")
        os.makedirs(root, exist_ok=True)
        self.register("storage_root", root)
        logs_dir = os.path.join(root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return root

    def _boot_model_registry(self) -> str:
        """Layer 3: Model registry — unified discovery across providers."""
        from app.model_registry import get_model_registry
        registry = get_model_registry()
        registry.discover_from_env()
        self.register("model_registry", registry)
        all_models = registry.list_models()
        self.register("registered_models", [m.name for m in all_models])
        return f"registered={len(all_models)}"

    def _boot_inference(self) -> str:
        """Layer 4: Inference — uses CloudConfig directly."""
        cfg = self._config
        self.register("inference_config", {
            "n_ctx": cfg.n_ctx,
            "n_threads": cfg.n_threads,
            "n_batch": cfg.n_batch,
            "n_ubatch": cfg.n_ubatch,
        })

        # Best-effort: start qwen_server.sh as a managed backend provider.
        try:
            from app.providers.qwen_server_provider import get_qwen_server_provider
            provider = get_qwen_server_provider()
            if provider.start():
                self.register("qwen_provider", provider)
                self.register_shutdown(provider.stop)
                logger.info("Qwen server provider started on %s:%d", provider.host, provider.port)
            else:
                logger.info("Qwen server provider not started (missing model/backend)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen server provider init failed: %s", exc)

        return f"n_ctx={cfg.n_ctx} n_threads={cfg.n_threads} n_batch={cfg.n_batch} n_ubatch={cfg.n_ubatch}"

    def _boot_embeddings(self) -> str:
        """Layer 4b: Embedding service (best-effort)."""
        try:
            from app.cognitive_gateway.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
            self.register("embedding_service", embedding_service)
            return "embedding_service_initialized"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding service init skipped: %s", exc)
            return f"embedding_service_skipped: {exc}"

    def _boot_vector_database(self) -> str:
        """Layer 5: Vector database (best-effort)."""
        try:
            from app.cognitive_gateway.vector_store import VectorStore
            vs = VectorStore()
            self.register("vector_db", vs)
            self.register("vector_store", vs)
            return "vector_db_initialized"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector database init skipped: %s", exc)
            return f"vector_db_skipped: {exc}"

    def _boot_api_services(self) -> str:
        """Layer 6: API services and cognitive subsystems."""
        self._initialize_cognitive_subsystems()
        self._initialize_providers()
        host = self._config.host
        port = self._config.port
        self.register("api_host", host)
        self.register("api_port", port)
        return f"{host}:{port}"

    def _initialize_providers(self) -> None:
        """Discover and start inference providers."""
        try:
            from app.providers.registry import ProviderRegistry
            registry = ProviderRegistry()
            registry.discover_from_env()
            registry.start_all()
            self.register("provider_registry", registry)
            logger.info("Provider registry initialized with %d providers (id=%d)",
                        len(registry.list_providers()), id(registry))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Provider registry init failed: %s", exc)

    def _initialize_cognitive_subsystems(self) -> None:
        """Initialize all cognitive subsystems with proper dependency ordering."""
        try:
            from app.event_bus import get_event_bus
            event_bus = get_event_bus()
            self.register("event_bus", event_bus)
            logger.info("Event bus initialized")
        except Exception as exc:
            logger.warning("Event bus init failed: %s", exc)

        try:
            from app.cognitive_envelope import normalize_envelope
            self.register("cognitive_envelope", normalize_envelope)
            logger.info("Cognitive envelope initialized")
        except Exception as exc:
            logger.warning("Cognitive envelope init failed: %s", exc)

        try:
            from app.node_identity import get_node_identity
            node_identity = get_node_identity()
            self.register("node_identity", node_identity)
            logger.info("Node identity initialized")
        except Exception as exc:
            logger.warning("Node identity init failed: %s", exc)

        try:
            from app.cloud_governance import get_cloud_governance
            governance = get_cloud_governance()
            self.register("cloud_governance", governance)
            logger.info("Cloud governance initialized")
        except Exception as exc:
            logger.warning("Cloud governance init failed: %s", exc)

        try:
            from app.temporal_sync import get_temporal_sync
            temporal_sync = get_temporal_sync()
            self.register("temporal_sync", temporal_sync)
            logger.info("Temporal sync initialized")
        except Exception as exc:
            logger.warning("Temporal sync init failed: %s", exc)

        try:
            from app.attention_allocator import get_attention_allocator
            attention_allocator = get_attention_allocator()
            self.register("attention_allocator", attention_allocator)
            logger.info("Attention allocator initialized")
        except Exception as exc:
            logger.warning("Attention allocator init failed: %s", exc)

        try:
            from app.reflection_engine import get_reflection_engine
            reflection_engine = get_reflection_engine()
            self.register("reflection_engine", reflection_engine)
            logger.info("Reflection engine initialized")
        except Exception as exc:
            logger.warning("Reflection engine init failed: %s", exc)

        try:
            from app.model_orchestrator import get_model_orchestrator
            model_ids = self._state.get("registered_models", [])
            orchestrator = get_model_orchestrator(model_ids=model_ids)
            self.register("model_orchestrator", orchestrator)
            logger.info("Model orchestrator initialized with %d models", len(model_ids))
        except Exception as exc:
            logger.warning("Model orchestrator init failed: %s", exc)

        try:
            from app.trading_runtime_bridge import get_trading_bridge
            trading_bridge = get_trading_bridge()
            self.register("trading_bridge", trading_bridge)
            logger.info("Trading runtime bridge initialized")
        except Exception as exc:
            logger.warning("Trading runtime bridge init failed: %s", exc)

        try:
            from app.federation import get_federation_manager
            federation_manager = get_federation_manager()
            self.register("federation_manager", federation_manager)
            logger.info("Federation manager initialized")
        except Exception as exc:
            logger.warning("Federation manager init failed: %s", exc)

    def _boot_websocket(self) -> str:
        return "websocket_deferred"

    def _boot_scheduler(self) -> str:
        return "scheduler_deferred"

    def _boot_background_services(self) -> str:
        try:
            self._start_background_workers()
            return "background_services_started"
        except Exception as exc:
            logger.warning("Background services init failed: %s", exc)
            return f"background_services_partial: {exc}"

    def _start_background_workers(self) -> None:
        try:
            from app.scheduler import get_scheduler
            scheduler = get_scheduler()
            self.register("scheduler", scheduler)
            scheduler.start()
            self.register_shutdown(scheduler.stop)
            logger.info("Scheduler started")
        except Exception as exc:
            logger.info("Scheduler not available: %s", exc)
        try:
            from app.workers import get_worker_manager
            worker_manager = get_worker_manager()
            self.register("worker_manager", worker_manager)
            worker_manager.start()
            self.register_shutdown(worker_manager.stop)
            logger.info("Worker manager started")
        except Exception as exc:
            logger.info("Worker manager not available: %s", exc)

    def _boot_plugins(self) -> str:
        plugins_dir = self._config.plugins_dir or ""
        if not plugins_dir:
            return "plugins_disabled"
        if not os.path.isdir(plugins_dir):
            return f"plugins_dir_missing: {plugins_dir}"
        loaded = 0
        for filename in os.listdir(plugins_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                loaded += 1
        return f"plugins_found={loaded}"

    def _boot_market_data(self) -> str:
        """Layer 11: Market Data Adapter initialization."""
        try:
            from app.trading.market_adapter import get_market_data_adapter
            adapter = get_market_data_adapter()
            self.register("market_data_adapter", adapter)
            return f"market_data_initialized symbols={len(adapter.get_symbols())}"
        except Exception as exc:
            logger.warning("Market data adapter init failed: %s", exc)
            return f"market_data_skipped: {exc}"

    def _boot_market_state(self) -> str:
        """Layer 12: Market Intelligence Engine initialization."""
        try:
            from app.trading.market_intelligence import get_market_intelligence
            engine = get_market_intelligence()
            self.register("market_intelligence", engine)
            return "market_intelligence_initialized"
        except Exception as exc:
            logger.warning("Market intelligence init failed: %s", exc)
            return f"market_intelligence_skipped: {exc}"

    def _boot_memory(self) -> str:
        """Layer 13: Memory integration and cognitive feedback loop initialization."""
        try:
            from app.trading.execution_feedback import get_execution_feedback
            feedback = get_execution_feedback()
            self.register("execution_feedback", feedback)
            logger.info("Execution feedback initialized")
        except Exception as exc:
            logger.warning("Execution feedback init failed: %s", exc)

        try:
            from app.trading.reflection_engine import ReflectionEngine
            reflection_engine = ReflectionEngine()
            self.register("reflection_engine", reflection_engine)
            logger.info("Reflection engine initialized")
        except Exception as exc:
            logger.warning("Reflection engine init failed: %s", exc)

        try:
            from app.trading.strategy_evolution import StrategyEvolutionEngine
            strategy_evolution = StrategyEvolutionEngine()
            self.register("strategy_evolution", strategy_evolution)
            logger.info("Strategy evolution engine initialized")
        except Exception as exc:
            logger.warning("Strategy evolution init failed: %s", exc)

        try:
            from app.trading.evaluation_engine import EvaluationEngine
            evaluation_engine = EvaluationEngine()
            self.register("evaluation_engine", evaluation_engine)
            logger.info("Evaluation engine initialized")
        except Exception as exc:
            logger.warning("Evaluation engine init failed: %s", exc)

        try:
            from app.trading.policy_optimizer import PolicyOptimizer
            policy_optimizer = PolicyOptimizer()
            self.register("policy_optimizer", policy_optimizer)
            logger.info("Policy optimizer initialized")
        except Exception as exc:
            logger.warning("Policy optimizer init failed: %s", exc)

        try:
            from app.trading.autonomous_learning_engine import AutonomousLearningEngine
            ale = AutonomousLearningEngine(
                evaluation_engine=self.get_service("evaluation_engine"),
                policy_optimizer=self.get_service("policy_optimizer"),
                reflection_engine=self.get_service("reflection_engine"),
                strategy_evolution=self.get_service("strategy_evolution"),
            )
            self.register("autonomous_learning_engine", ale)
            logger.info("Autonomous learning engine initialized")
        except Exception as exc:
            logger.warning("Autonomous learning engine init failed: %s", exc)

        return "memory_cognitive_feedback_initialized"

    def _boot_lean_algorithms(self) -> str:
        """Layer 14: Lean-Algos bridge and LeanAlgoManager initialization."""
        try:
            from app.trading.lean_bridge import get_lean_bridge
            bridge = get_lean_bridge()
            self.register("lean_bridge", bridge)
            logger.info("Lean bridge initialized")
        except Exception as exc:
            logger.warning("Lean bridge init failed: %s", exc)

        try:
            from app.trading.lean_algo_manager import LeanAlgoManager
            lean_manager = LeanAlgoManager()
            lean_manager.initialize()
            self.register("lean_algo_manager", lean_manager)
            logger.info("LeanAlgoManager initialized")
        except Exception as exc:
            logger.warning("LeanAlgoManager init failed: %s", exc)

        return "lean_algorithms_initialized"

    def _boot_llm_validation(self) -> str:
        """Layer 15: GGUF validation initialization."""
        try:
            from app.trading.gguf_validator import get_gguf_validator
            validator = get_gguf_validator()
            self.register("gguf_validator", validator)
            return "gguf_validator_initialized"
        except Exception as exc:
            logger.warning("GGUF validator init failed: %s", exc)
            return f"gguf_validator_skipped: {exc}"

    def _boot_risk_engine(self) -> str:
        """Layer 16: Risk engine initialization."""
        try:
            from app.trading.risk_engine import get_risk_engine
            engine = get_risk_engine()
            self.register("risk_engine", engine)
            return "risk_engine_initialized"
        except Exception as exc:
            logger.warning("Risk engine init failed: %s", exc)
            return f"risk_engine_skipped: {exc}"

    def _boot_freqtrade(self) -> str:
        """Layer 17: Freqtrade adapter initialization."""
        try:
            from app.trading.freqtrade_adapter import get_freqtrade_adapter
            adapter = get_freqtrade_adapter()
            self.register("freqtrade_adapter", adapter)
            return f"freqtrade_adapter_initialized enabled={adapter.enabled}"
        except Exception as exc:
            logger.warning("Freqtrade adapter init failed: %s", exc)
            return f"freqtrade_adapter_skipped: {exc}"

    def _boot_execution_monitor(self) -> str:
        """Layer 18: Execution monitor initialization with pipeline."""
        try:
            from app.trading.pipeline import get_trading_pipeline
            pipeline = get_trading_pipeline()
            self.register("trading_pipeline", pipeline)
            return "trading_pipeline_initialized"
        except Exception as exc:
            logger.warning("Trading pipeline init failed: %s", exc)
            return f"trading_pipeline_skipped: {exc}"

    def _boot_health(self) -> str:
        """Layer 19: Health monitor — uses shared ModelManager from config."""
        from app.main import ModelManager
        cfg = self._config
        models = cfg.model_map
        default_model = cfg.default_model
        # Create ModelManager from the SAME CloudConfig so it shares the same model list.
        manager = ModelManager(models, default_model, config=cfg)
        self.register("model_manager", manager)

        health_status = self._verify_health()
        self.register("health_status", health_status)
        degraded = any(s.get("status") != "ok" for s in health_status.values() if isinstance(s, dict))
        if degraded:
            logger.warning("Health verification shows degraded subsystems")
        return "health_verified"

    def _verify_health(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        try:
            manager = self.get_service("model_manager")
            models = manager.list_models() if manager else []
            status["models"] = {"status": "ok", "count": len(models)}
        except Exception as exc:
            status["models"] = {"status": "error", "detail": str(exc)}
        status["api"] = {"status": "ok", "endpoint": f"{self._state.get('api_host')}:{self._state.get('api_port')}"}
        status["websocket"] = {"status": "deferred"}
        if self.get_service("vector_db"):
            status["vector_db"] = {"status": "ok"}
        else:
            status["vector_db"] = {"status": "skipped"}
        status["storage"] = {"status": "ok", "root": self._state.get("storage_root")}
        scheduler = self.get_service("scheduler")
        status["scheduler"] = {"status": "ok" if scheduler else "deferred"}
        status["inference"] = {"status": "ok", "config": self._state.get("inference_config")}
        return status

    def _finalize(self) -> bool:
        failed = [s for s in self._stages if s.status == "failed"]
        ready = len(failed) == 0
        self._stages.append(
            StageResult(
                stage=BootStage.READY if ready else BootStage.DEGRADED,
                status="completed" if ready else "degraded",
                message=f"stages_failed={len(failed)}",
            )
        )
        status = "READY" if ready else "DEGRADED"
        logger.info("Cloud %s", status)
        for stage in self._stages:
            logger.info(
                "  [%s] %s (%.1f ms) %s",
                stage.stage.value,
                stage.status,
                stage.duration_ms,
                stage.message,
            )
        return ready


def boot_runtime(config: Any | None = None) -> CloudRuntime:
    """Create and run the CloudRuntime. Returns the runtime instance."""
    runtime = CloudRuntime(config=config)
    ready = runtime.run()
    if not ready:
        logger.warning("CloudRuntime boot completed in degraded mode")
    return runtime