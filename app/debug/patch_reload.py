"""Patch script: adds _reload_model method to GGUFEngine class."""
import pathlib

f = pathlib.Path("Niblit-cloud-server/app/main.py")
content = f.read_text()

# Insert _reload_model method before the @property metadata property
anchor = "    @property\n    def metadata(self) -> dict[str, Any]:"

method = '''    def _reload_model(self) -> None:
        """Reload the GGUF model to recover from native state corruption.

        Called when llama_decode returns -1 (GGML_ASSERT failure), which
        indicates the native llama.cpp context has been corrupted by a
        previous large generation. Reloading creates a fresh context.
        """
        model_path = self._model_meta["file"]
        n_ctx = self._model_meta["n_ctx"]
        n_threads = self._model_meta["n_threads"]
        logger.info("Reloading model: %s", model_path)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is required.") from exc
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "use_mmap": False,
            "use_mlock": False,
            "seed": 42,
            "verbose": False,
        }
        gpu_layers = os.getenv("NIBLIT_N_GPU_LAYERS", "").strip()
        if gpu_layers:
            try:
                kwargs["n_gpu_layers"] = int(gpu_layers)
            except (TypeError, ValueError):
                pass
        self._llm = Llama(**kwargs)
        logger.info("Model reloaded successfully: %s", model_path)

'''

if anchor in content:
    content = content.replace(anchor, method + anchor, 1)
    f.write_text(content)
    print("PATCHED: _reload_model method added")
else:
    print("ANCHOR NOT FOUND")
