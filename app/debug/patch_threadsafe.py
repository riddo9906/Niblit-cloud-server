"""Patch script: adds thread safety to GGUFEngine inference.

Root cause: llama-cpp-python 0.3.16's Llama object is NOT thread-safe.
Concurrent create_chat_completion() calls on the same Llama instance
cause native memory corruption → llama_decode returned -1 → GGML_ASSERT → segfault.

Fix: Add a threading.Lock to GGUFEngine that serializes all inference calls.
Also fix ModelManager.chat() race condition where engines dict is accessed
without the lock.
"""
import pathlib

f = pathlib.Path("Niblit-cloud-server/app/main.py")
content = f.read_text()

# 1. Add _lock to GGUFEngine.__init__
old_init_end = '''        self._model_meta: dict[str, Any] = {
            "file": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
        }
        logger.info("Model loaded successfully from %s", model_path)'''

new_init_end = '''        self._model_meta: dict[str, Any] = {
            "file": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
        }
        self._lock = threading.Lock()
        logger.info("Model loaded successfully from %s", model_path)'''

if old_init_end in content:
    content = content.replace(old_init_end, new_init_end, 1)
    print("PATCHED: Added _lock to GGUFEngine.__init__")
else:
    print("SKIP: GGUFEngine.__init__ lock already present or not found")

# 2. Wrap chat() body in lock
old_chat_start = '''    def chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        self.validate_prompt(messages, max_tokens)

        safe_messages: list[dict[str, str]] = []
        for msg in messages:
            safe_messages.append({
                "role": str(msg.get("role", "user")).strip(),
                "content": str(msg.get("content", " ")).strip() or " ",
            })

        logger.info(
            "Inference: messages=%d prompt~%d max_tokens=%d temp=%.2f",
            len(safe_messages), self._estimate_tokens_rough(safe_messages),
            max_tokens, temperature,
        )

        try:
            response = self._llm.create_chat_completion('''

new_chat_start = '''    def chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        self.validate_prompt(messages, max_tokens)

        safe_messages: list[dict[str, str]] = []
        for msg in messages:
            safe_messages.append({
                "role": str(msg.get("role", "user")).strip(),
                "content": str(msg.get("content", " ")).strip() or " ",
            })

        logger.info(
            "Inference: messages=%d prompt~%d max_tokens=%d temp=%.2f",
            len(safe_messages), self._estimate_tokens_rough(safe_messages),
            max_tokens, temperature,
        )

        # llama-cpp-python 0.3.16 Llama object is NOT thread-safe.
        # Serialize all create_chat_completion() calls to prevent
        # concurrent native access that causes llama_decode returned -1
        # and GGML_ASSERT failures.
        with self._lock:
            try:
                response = self._llm.create_chat_completion('''

if old_chat_start in content:
    content = content.replace(old_chat_start, new_chat_start, 1)
    print("PATCHED: Wrapped chat() inference in lock")
else:
    print("SKIP: chat() lock wrapper already present or not found")

# 3. Fix indentation of the rest of chat() body (the except blocks and return)
# After wrapping in 'with self._lock:', everything inside needs +4 indentation
# We need to find the end of the try/except block and the return statement

# Find the section from "except RuntimeError" to the end of chat() method
old_chat_rest = '''        except RuntimeError as exc:
            exc_str = str(exc)
            logger.error("llama.cpp error: %s", exc_str)
            if "llama_decode returned" in exc_str or "GGML_ASSERT" in exc_str:
                logger.info("Attempting model reload after decode error...")
                try:
                    self._reload_model()
                    response = self._llm.create_chat_completion(
                        messages=safe_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                except Exception as retry_exc:
                    logger.error("Retry after reload failed: %s", retry_exc)
                    raise HTTPException(
                        status_code=503,
                        detail={"error": "inference_failed", "message": str(retry_exc)},
                    ) from retry_exc
            else:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "inference_failed", "message": exc_str},
                ) from exc
        except ValueError as exc:
            logger.error("llama.cpp value error: %s", exc)
            raise HTTPException(status_code=503, detail={"error": "inference_failed", "message": str(exc)}) from exc
        except MemoryError as exc:
            logger.error("llama.cpp OOM: %s", exc)
            raise HTTPException(status_code=503, detail={"error": "out_of_memory", "message": str(exc)}) from exc

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise HTTPException(status_code=502, detail={"error": "empty_response", "message": "Model returned empty choices."})
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise HTTPException(status_code=502, detail={"error": "invalid_response", "message": "message.content must be a string."})
        usage = response.get("usage")
        logger.info("Inference OK: finish=%s tokens=%s", choice.get("finish_reason"), usage.get("total_tokens") if usage else "?")
        return ModelEngineResult(
            text=text,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=usage,
        )'''

new_chat_rest = '''            except RuntimeError as exc:
                exc_str = str(exc)
                logger.error("llama.cpp error: %s", exc_str)
                if "llama_decode returned" in exc_str or "GGML_ASSERT" in exc_str:
                    logger.info("Attempting model reload after decode error...")
                    try:
                        self._reload_model()
                        response = self._llm.create_chat_completion(
                            messages=safe_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False,
                        )
                    except Exception as retry_exc:
                        logger.error("Retry after reload failed: %s", retry_exc)
                        raise HTTPException(
                            status_code=503,
                            detail={"error": "inference_failed", "message": str(retry_exc)},
                        ) from retry_exc
                else:
                    raise HTTPException(
                        status_code=503,
                        detail={"error": "inference_failed", "message": exc_str},
                    ) from exc
            except ValueError as exc:
                logger.error("llama.cpp value error: %s", exc)
                raise HTTPException(status_code=503, detail={"error": "inference_failed", "message": str(exc)}) from exc
            except MemoryError as exc:
                logger.error("llama.cpp OOM: %s", exc)
                raise HTTPException(status_code=503, detail={"error": "out_of_memory", "message": str(exc)}) from exc

            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise HTTPException(status_code=502, detail={"error": "empty_response", "message": "Model returned empty choices."})
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise HTTPException(status_code=502, detail={"error": "invalid_response", "message": "message.content must be a string."})
            usage = response.get("usage")
            logger.info("Inference OK: finish=%s tokens=%s", choice.get("finish_reason"), usage.get("total_tokens") if usage else "?")
            return ModelEngineResult(
                text=text,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
            )'''

if old_chat_rest in content:
    content = content.replace(old_chat_rest, new_chat_rest, 1)
    print("PATCHED: Indented chat() body inside lock")
else:
    print("SKIP: chat() body indentation already done or not found")

# 4. Fix _reload_model to acquire lock before replacing self._llm
old_reload = '''        self._llm = Llama(**kwargs)
        logger.info("Model reloaded successfully: %s", model_path)'''

new_reload = '''        with self._lock:
            self._llm = Llama(**kwargs)
        logger.info("Model reloaded successfully: %s", model_path)'''

if old_reload in content:
    content = content.replace(old_reload, new_reload, 1)
    print("PATCHED: _reload_model acquires lock")
else:
    print("SKIP: _reload_model lock already present or not found")

# 5. Fix ModelManager.chat() race condition - access engines under lock
old_mm_chat = '''        if model_id not in self._engines:
            self._engines[model_id] = GGUFEngine(
                model_path=self._model_map[model_id],
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                runtime_options=self._runtime_options,
            )
        plan = self._prepare_inference(messages=messages, max_tokens=max_tokens)'''

new_mm_chat = '''        with self._lock:
            if model_id not in self._engines:
                self._engines[model_id] = GGUFEngine(
                    model_path=self._model_map[model_id],
                    n_ctx=self._n_ctx,
                    n_threads=self._n_threads,
                    runtime_options=self._runtime_options,
                )
            engine = self._engines[model_id]
        plan = self._prepare_inference(messages=messages, max_tokens=max_tokens)'''

if old_mm_chat in content:
    content = content.replace(old_mm_chat, new_mm_chat, 1)
    print("PATCHED: ModelManager.chat() engine access under lock")
else:
    print("SKIP: ModelManager.chat() lock already present or not found")

# 6. Fix ModelManager.chat() to use local engine reference
old_mm_chat_call = '''        try:
            result = self._engines[model_id].chat('''

new_mm_chat_call = '''        try:
            result = engine.chat('''

if old_mm_chat_call in content:
    content = content.replace(old_mm_chat_call, new_mm_chat_call, 1)
    print("PATCHED: ModelManager.chat() uses local engine reference")
else:
    print("SKIP: ModelManager.chat() engine reference already fixed or not found")

f.write_text(content)
print("\nAll patches applied.")
