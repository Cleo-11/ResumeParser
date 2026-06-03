"""
core/logger.py
Structured logger used across all pipeline layers.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List

COLORS = {
    "DEBUG":   "\033[90m",
    "INFO":    "\033[36m",
    "SUCCESS": "\033[32m",
    "WARN":    "\033[33m",
    "ERROR":   "\033[31m",
    "RESET":   "\033[0m",
    "BOLD":    "\033[1m",
}

_handlers: List[Callable[[Dict], None]] = []


def add_handler(fn: Callable[[Dict], None]) -> None:
    """Register a callback that receives every log event dict."""
    _handlers.append(fn)


def _emit(level: str, layer: str, msg: str, **extra: Any) -> None:
    event = {
        "ts":    datetime.now(timezone.utc).isoformat(),
        "level": level,
        "layer": layer,
        "msg":   msg,
        **extra,
    }
    for h in _handlers:
        try:
            h(event)
        except Exception:
            pass

    c = COLORS.get(level, "")
    r = COLORS["RESET"]
    b = COLORS["BOLD"]
    tag = f"[{layer}]"
    print(f"{c}{b}{level:<8}{r} {b}{tag:<18}{r} {msg}", file=sys.stderr)


class LayerLogger:
    """Logger scoped to a single pipeline layer."""

    def __init__(self, layer: str):
        self.layer = layer
        self._start = time.perf_counter()

    def debug(self, msg: str, **kw: Any):   _emit("DEBUG",   self.layer, msg, **kw)
    def info(self,  msg: str, **kw: Any):   _emit("INFO",    self.layer, msg, **kw)
    def success(self, msg: str, **kw: Any): _emit("SUCCESS", self.layer, msg, **kw)
    def warn(self,  msg: str, **kw: Any):   _emit("WARN",    self.layer, msg, **kw)
    def error(self, msg: str, **kw: Any):   _emit("ERROR",   self.layer, msg, **kw)

    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 2)