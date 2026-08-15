"""Qwen Server Provider — manages qwen_server.sh as a managed backend."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QwenServerProvider:
    """Manages the qwen_server.sh backend process."""

    model_path: str = ""
    backend_bin: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    n_ctx: int = 32768
    n_threads: int = 8
    n_gpu_layers: int = 35
    process: subprocess.Popen | None = None
    pid_file: str = ""
    log_file: str = ""
    ready: bool = False
    start_time: float = field(default_factory=time.time)

    def start(self) -> bool:
        """Start the qwen server backend."""
        if self.process is not None:
            return True

        self.model_path = self.model_path or os.getenv("NIBLIT_MODEL_PATH", "")
        self.backend_bin = self.backend_bin or os.getenv("NIBLIT_LLAMA_SERVER_BIN", "")
        self.host = os.getenv("NIBLIT_HOST", "127.0.0.1")
        self.port = int(os.getenv("NIBLIT_PORT", "8000"))
        self.n_ctx = int(os.getenv("NIBLIT_N_CTX", "32768"))
        self.n_threads = int(os.getenv("NIBLIT_N_THREADS", "8"))
        self.n_gpu_layers = int(os.getenv("NIBLIT_N_GPU_LAYERS", "35"))

        if not self.model_path or not os.path.isfile(self.model_path):
            logger.warning("QwenServerProvider: model not found at %s", self.model_path)
            return False

        if not self.backend_bin or not os.path.isfile(self.backend_bin):
            logger.warning("QwenServerProvider: backend binary not found at %s", self.backend_bin)
            return False

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tools_dir = os.path.join(script_dir, "tools")
        qwen_script = os.path.join(tools_dir, "qwen_server.sh")

        if not os.path.isfile(qwen_script):
            logger.warning("QwenServerProvider: qwen_server.sh not found at %s", qwen_script)
            return False

        self.pid_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), "niblit-qwen-server.pid")
        self.log_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), "niblit-qwen-server.log")

        env = {**os.environ}
        env.update({
            "NIBLIT_MODEL_PATH": self.model_path,
            "NIBLIT_LLAMA_SERVER_BIN": self.backend_bin,
            "NIBLIT_HOST": self.host,
            "NIBLIT_PORT": str(self.port),
            "NIBLIT_N_CTX": str(self.n_ctx),
            "NIBLIT_N_THREADS": str(self.n_threads),
            "NIBLIT_N_GPU_LAYERS": str(self.n_gpu_layers),
            "NIBLIT_TUNNEL_PROVIDER": "none",
        })

        try:
            with open(self.log_file, "a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting qwen_server.sh\n")

            self.process = subprocess.Popen(
                ["bash", qwen_script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            if self.process.pid:
                with open(self.pid_file, "w", encoding="utf-8") as pid_fh:
                    pid_fh.write(str(self.process.pid))

            logger.info("QwenServerProvider started: pid=%d port=%d", self.process.pid, self.port)
            self.ready = True
            return True

        except Exception as exc:
            logger.error("QwenServerProvider failed to start: %s", exc)
            self.process = None
            return False

    def stop(self) -> None:
        """Stop the qwen server backend."""
        if self.process is None:
            return

        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            logger.info("QwenServerProvider stopped")
        except Exception as exc:
            logger.warning("QwenServerProvider stop error: %s", exc)
        finally:
            self.process = None
            self.ready = False

        if self.pid_file and os.path.isfile(self.pid_file):
            try:
                os.unlink(self.pid_file)
            except OSError:
                pass

    def status(self) -> dict[str, Any]:
        """Return provider health status."""
        if self.process is None:
            return {"status": "stopped", "ready": False}

        return_code = self.process.poll()
        if return_code is not None:
            self.ready = False
            return {"status": "error", "ready": False, "return_code": return_code}

        return {
            "status": "running",
            "ready": self.ready,
            "pid": self.process.pid,
            "port": self.port,
            "model_path": self.model_path,
            "backend_bin": self.backend_bin,
        }


def get_qwen_server_provider() -> QwenServerProvider:
    """Return the process-level QwenServerProvider singleton."""
    global _qwen_provider
    try:
        return _qwen_provider
    except NameError:
        _qwen_provider = QwenServerProvider()
        return _qwen_provider