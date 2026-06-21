from __future__ import annotations

import os
import shutil
from pathlib import Path


def _prefer_git_bash_on_windows() -> None:
    if os.name != "nt":
        return

    os.environ.setdefault("MSYS2_PATH_TYPE", "inherit")
    os.environ.setdefault("MSYS_NO_PATHCONV", "1")

    candidates = [
        Path(r"C:\Program Files\Git\bin"),
        Path(r"C:\Program Files (x86)\Git\bin"),
    ]
    for candidate in candidates:
        bash_path = candidate / "bash.exe"
        if bash_path.exists():
            current = os.environ.get("PATH", "")
            candidate_str = str(candidate)
            if candidate_str not in current.split(os.pathsep):
                os.environ["PATH"] = candidate_str + os.pathsep + current
            break


_prefer_git_bash_on_windows()
