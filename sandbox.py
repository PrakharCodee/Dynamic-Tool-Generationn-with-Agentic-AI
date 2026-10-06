"""
sandbox.py — Optimized Sandbox with pre-loaded C-extensions.

This version pre-loads 'cv2' and 'numpy' to bypass local import errors
that can occur with C-extension hooks in sandboxed environments.
"""

import builtins
import io
import sys
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

# ───────────────────────── allow / block lists ─────────────────────────

ALLOWED_MODULES = {
    "numpy", "math", "PIL", "PIL.Image", "PIL.ImageDraw",
    "PIL.ImageFilter", "PIL.ImageFont", "cv2",
    "skimage", "skimage.measure", "skimage.morphology",
    "skimage.filters", "skimage.color", "skimage.feature",
    "skimage.draw", "skimage.transform",
    "matplotlib", "matplotlib.pyplot",
    "collections", "statistics", "re", "json",
    "functools", "itertools", "copy", "typing",
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "requests", "shutil",
    "importlib", "ctypes", "pickle", "pathlib", "http", "urllib",
    "ftplib", "smtplib", "telnetlib", "webbrowser", "code",
    "codeop", "compileall", "py_compile", "zipfile", "tarfile",
    "tempfile", "glob", "signal", "multiprocessing", "threading",
    "concurrent", "_thread", "asyncio", "sqlite3",
}

BLOCKED_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open",
    "input", "breakpoint", "exit", "quit", "help",
}

# ───────────────────────── safe import ─────────────────────────

_original_import = builtins.__import__

def _safe_import(name, *args, **kwargs):
    """Import hook that enforces the module allowlist."""
    top = name.split(".")[0]
    if name in BLOCKED_MODULES or top in BLOCKED_MODULES:
        raise ImportError(f"Module '{name}' is blocked in sandbox")
    if name in ALLOWED_MODULES or top in ALLOWED_MODULES:
        try:
            return _original_import(name, *args, **kwargs)
        except ImportError:
            # Fallback for complex C-extensions already in globs
            return _original_import(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' is not in the sandbox allowlist")

# ───────────────────────── result dataclass ─────────────────────────

@dataclass
class SandboxResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

# ───────────────────────── public API ─────────────────────────

def execute(code_str: str, image: Image.Image,
            timeout: float = 5.0) -> SandboxResult:
    """Execute code in a restricted sandbox with pre-loaded tools."""
    if not code_str or not code_str.strip():
        return SandboxResult(success=False, exception="Empty code string")

    # Pre-import key modules to ensure availability
    import math, collections, re, json
    try:
        import cv2
    except ImportError:
        cv2 = None

    safe_bi = {}
    for name in dir(builtins):
        if name not in BLOCKED_BUILTINS and not name.startswith("_"):
            safe_bi[name] = getattr(builtins, name)
    safe_bi["__import__"] = _safe_import

    result_dict: Dict[str, Any] = {}
    
    globs = {
        "__builtins__": safe_bi,
        "image": image.copy(),
        "result": result_dict,
        "np": np,
        "numpy": np,
        "math": math,
        "collections": collections,
        "re": re,
        "json": json,
        "cv2": cv2,
        "PIL": __import__("PIL"),
        "Image": Image,
    }

    container: Dict[str, Any] = {
        "stdout": "", "stderr": "", "exception": None, "done": False,
    }

    def _run():
        out, err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = out, err
            # Note: Putting globs as both globals AND locals
            exec(code_str, globs, globs)
            container["stdout"] = out.getvalue()
            container["stderr"] = err.getvalue()
        except Exception as exc:
            container["exception"] = f"{type(exc).__name__}: {exc}"
            container["stdout"] = out.getvalue()
            container["stderr"] = err.getvalue() + "\n" + traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            container["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not container["done"]:
        return SandboxResult(success=False, exception=f"Timed out ({timeout}s)")
    if container["exception"]:
        return SandboxResult(success=False, stdout=container["stdout"], 
                             stderr=container["stderr"], result=result_dict, 
                             exception=container["exception"])

    return SandboxResult(success=True, stdout=container["stdout"], 
                         stderr=container["stderr"], result=result_dict)
