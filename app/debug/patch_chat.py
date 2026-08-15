"""Patch script: adds model reload on llama_decode error to GGUFEngine.chat()."""
import pathlib

f = pathlib.Path("Niblit-cloud-server/app/main.py")
content = f.read_text()

old = '''        except RuntimeError as exc:
            logger.error("llama.cpp error: %s", exc)
            raise HTTPException(status_code=503, detail={"error": "inference_failed", "message": str(exc)}) from exc
        except ValueError as exc:
            logger.error("llama.cpp value error: %s", exc)
            raise HTTPException(status_code=503, detail={"error": "inference_failed", "message": str(exc)}) from exc
        except MemoryError as exc:
            logger.error("llama.cpp OOM: %s", exc)
            raise HTTPException(status_code=503, detail={"error": "out_of_memory", "message": str(exc)}) from exc'''

new = '''        except RuntimeError as exc:
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
            raise HTTPException(status_code=503, detail={"error": "out_of_memory", "message": str(exc)}) from exc'''

if old in content:
    content = content.replace(old, new, 1)
    f.write_text(content)
    print("PATCHED: chat() error handling with model reload")
else:
    print("OLD BLOCK NOT FOUND")
