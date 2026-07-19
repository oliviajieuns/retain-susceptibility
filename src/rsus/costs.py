"""Cost telemetry. Every scorer and stage returns a CostRecord; the RQ6
cost tables are generated from these records, never hand-filled."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class CostRecord:
    wall_s: float = 0.0
    peak_mem_bytes: int = 0
    fwd_passes: int = 0
    bwd_passes: int = 0
    tokens_fwd: int = 0
    tokens_bwd: int = 0
    notes: dict = field(default_factory=dict)

    def merge(self, other: "CostRecord") -> "CostRecord":
        return CostRecord(
            wall_s=self.wall_s + other.wall_s,
            peak_mem_bytes=max(self.peak_mem_bytes, other.peak_mem_bytes),
            fwd_passes=self.fwd_passes + other.fwd_passes,
            bwd_passes=self.bwd_passes + other.bwd_passes,
            tokens_fwd=self.tokens_fwd + other.tokens_fwd,
            tokens_bwd=self.tokens_bwd + other.tokens_bwd,
            notes={**self.notes, **other.notes},
        )


class Meter:
    """Context manager filling wall time and (if CUDA) peak memory into an
    attached CostRecord. Pass counters are incremented by the caller."""

    def __init__(self, rec: CostRecord):
        self.rec = rec

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self.rec

    def __exit__(self, *exc):
        self.rec.wall_s += time.perf_counter() - self._t0
        if torch.cuda.is_available():
            self.rec.peak_mem_bytes = max(
                self.rec.peak_mem_bytes, torch.cuda.max_memory_allocated()
            )
        return False
