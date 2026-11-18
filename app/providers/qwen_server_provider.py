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
        self.port = int(os.getenv("NIBLIT_QWEN_PORT", str(self.port)))
        self.n_ctx = int(os.getenv("NIBLIT_N_CTX", "32768"))
        self.n_threads = int(os.getenv("NIBLIT_N_THREADS", "8"))
        self.n_gpu_layers = int(os.getenv("NIBLIT_N_GPU_LAYERS", "35"))

        if not self.model_path or not os.path.isfile(self.model_path):
            logger.warning("QwenServerProvider: model not found at %s", self.model_path)
            return False

        if not self.backend_bin or not os.path.isfile(self.backend_bin):
            logger.warning("QwenServerProvider: backend binary not found at %s", self.backend_bin)
            return False

        if os.name == 'nt':
            return self._start_native_windows()

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

    def _start_native_windows(self, profile_override=None) -> bool:
        """Use the installed llama.cpp directly; no bash, downloads or shell."""
        import hashlib,json,socket
        from pathlib import Path
        import httpx
        profile_path=profile_override or os.environ.get('NIBLIT_ACTIVE_MODEL_PROFILE','')
        if not profile_path:
            logger.warning('Native Windows provider requires an explicit versioned profile')
            return False
        try:
            raw=Path(profile_path).read_bytes()
            if len(raw)>65536:raise ValueError('Model profile too large')
            profile=json.loads(raw)
            if profile.get('schema_version')!='niblit-local-model/1' or profile.get('architecture')!='qwen2':
                raise ValueError('Unsupported model architecture/profile')
            if profile.get('tokenizer')!='embedded-gguf' or profile.get('model_id')!='qwen2.5-coder-3b':
                raise ValueError('Incompatible tokenizer/model identity')
            version=profile.get('version')
            if not isinstance(version,str) or not version or len(version)>80:raise ValueError('Versioned model profile required')
            settings=profile.get('runtime',{})
            ctx=settings.get('context_length',self.n_ctx)
            threads=settings.get('threads',self.n_threads)
            gpu=settings.get('gpu_layers',self.n_gpu_layers)
            if any(isinstance(v,bool) or not isinstance(v,int) for v in (ctx,threads,gpu)):
                raise ValueError('Explicit integer runtime settings required')
            if not 1024<=ctx<=32768 or not 1<=threads<=max(1,os.cpu_count() or 1) or gpu!=0:
                raise ValueError('Unsupported local runtime/resource profile')
            self.n_ctx,self.n_threads,self.n_gpu_layers=ctx,threads,gpu
            self.profile_version=version
            for key,path in (('executable',self.backend_bin),('model',self.model_path)):
                if str(Path(path).resolve())!=str(Path(profile[key]['path']).resolve()):
                    raise ValueError('Model or runtime profile binding mismatch')
                with open(path,'rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual!=profile[key]['sha256']:raise ValueError('Model/runtime integrity mismatch')
            with open(self.model_path,'rb') as stream:
                if stream.read(4)!=b'GGUF':raise ValueError('GGUF model required')
            if self.host!='127.0.0.1' or not 1024<=self.port<=65535:raise ValueError('Loopback provider required')
            with socket.socket() as probe:
                if probe.connect_ex((self.host,self.port))==0:raise ValueError('Inference port already owned')
            keyfile=os.environ.get('NIBLIT_PORTAL_GGUF_KEY_FILE','')
            if not keyfile or len(Path(keyfile).read_text().strip())<32:raise ValueError('Native authentication required')
            logs=Path(os.environ.get('NIBLIT_SERVICE_LOG_DIR',os.environ.get('TMPDIR','.')))
            logs.mkdir(parents=True,exist_ok=True);self.log_file=str(logs/'llama-server.log')
            cmd=[self.backend_bin,'-m',self.model_path,'--host',self.host,'--port',str(self.port),
                 '--ctx-size',str(self.n_ctx),'--threads',str(self.n_threads),
                 '--n-gpu-layers',str(self.n_gpu_layers),'--parallel','1','--alias',profile['model_id'],
                 '--api-key-file',keyfile]
            with open(self.log_file,'ab') as log:
                self.process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),shell=False)
            deadline=time.monotonic()+120
            headers={'Authorization':'Bearer '+Path(keyfile).read_text().strip()}
            with httpx.Client(trust_env=False,timeout=2) as client:
                while time.monotonic()<deadline:
                    if self.process.poll() is not None:raise RuntimeError('Native provider exited before readiness')
                    try:
                        r=client.get(f'http://127.0.0.1:{self.port}/v1/models',headers=headers)
                        if r.status_code==200 and any(m.get('id')==profile['model_id'] for m in r.json().get('data',[])):
                            self.ready=True
                            return True
                    except httpx.HTTPError:pass
                    time.sleep(.25)
            raise TimeoutError('Native inference readiness timed out')
        except Exception as exc:
            logger.error('Native Windows provider unavailable: %s',type(exc).__name__)
            self.stop()
            fallback=os.environ.get('NIBLIT_FALLBACK_MODEL_PROFILE','')
            if profile_override is None and fallback and fallback!=profile_path:
                logger.warning('Experimental profile rejected; retaining configured known-good Qwen fallback')
                return self._start_native_windows(fallback)
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
            "profile_version": getattr(self,"profile_version",None),
            "context_length":self.n_ctx,
        }


def get_qwen_server_provider() -> QwenServerProvider:
    """Return the process-level QwenServerProvider singleton."""
    global _qwen_provider
    try:
        return _qwen_provider
    except NameError:
        _qwen_provider = QwenServerProvider()
        return _qwen_provider