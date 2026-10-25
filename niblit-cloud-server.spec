# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-folder build for the Niblit cloud server."""
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

REPO_ROOT = Path(SPECPATH).resolve()
APP_ROOT = REPO_ROOT / "app"
hiddenimports = collect_submodules("app")
hiddenimports += [
    "uvicorn", "uvicorn.main", "uvicorn.config", "uvicorn.logging",
    "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "fastapi", "starlette", "pydantic", "llama_cpp",
]
datas = collect_data_files("llama_cpp")
static_root = APP_ROOT / "static"
if static_root.is_dir():
    datas.append((str(static_root), "app/static"))
    # The frozen entry script executes as _internal/main.py, while imported
    # app.main executes under _internal/app/. Both use __file__ for assets.
    datas.append((str(static_root), "static"))
binaries = collect_dynamic_libs("llama_cpp")
llama_root = Path(__import__("llama_cpp").__file__).resolve().parent
llama_lib = llama_root / "lib"
if llama_lib.is_dir():
    binaries.extend((str(path), "llama_cpp/lib") for path in llama_lib.glob("*") if path.is_file())

# Optional native llama.cpp runtime. Resolve its source only at build time and
# place the executable beside its DLLs in the portable Cloud bundle. This does
# not alter the current GGUFEngine path, which uses llama-cpp-python.
native_root_value = os.environ.get("NIBLIT_LLAMA_CPP_ROOT", "").strip()
native_root = Path(native_root_value).expanduser() if native_root_value else Path.home() / "llama_migration" / "llama.cpp"
native_candidates = (
    native_root / "build" / "bin" / "Release",
    native_root / "build" / "bin",
)
native_bin_dir = next((candidate for candidate in native_candidates if (candidate / ("llama-server.exe" if os.name == "nt" else "llama-server")).is_file()), None)
if native_bin_dir is not None:
    native_exe = native_bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
    binaries.append((str(native_exe), "llama-runtime"))
    for native_dll in native_bin_dir.glob("*.dll"):
        binaries.append((str(native_dll), "llama-runtime"))
elif native_root_value:
    raise SystemExit(f"NIBLIT_LLAMA_CPP_ROOT has no built llama-server executable: {native_root}")

a = Analysis(
    [str(APP_ROOT / "main.py")], pathex=[str(REPO_ROOT)], binaries=binaries,
    datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={},
    runtime_hooks=[], excludes=["pytest", "_pytest", "IPython", "jupyter", "tkinter", "matplotlib"],
    noarchive=False, optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="niblit-cloud",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False,
    upx_exclude=[], name="cloud",
)
