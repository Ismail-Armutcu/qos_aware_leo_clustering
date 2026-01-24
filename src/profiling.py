# src/profiling.py
from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Dict

@dataclass
class Profiler:
    t: Dict[str, float] = field(default_factory=dict)
    c: Dict[str, int] = field(default_factory=dict)
    _start: Dict[str, float] = field(default_factory=dict)

    def tic(self, name: str) -> None:
        self._start[name] = time.perf_counter()

    def toc(self, name: str) -> None:
        dt = time.perf_counter() - self._start.get(name, time.perf_counter())
        self.t[name] = self.t.get(name, 0.0) + dt

    def inc(self, name: str, k: int = 1) -> None:
        self.c[name] = self.c.get(name, 0) + k
