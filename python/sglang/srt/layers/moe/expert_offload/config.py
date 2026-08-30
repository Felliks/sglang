from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpertOffloadConfig:
    """Validated, hardware-neutral budgets for one expert-offload runtime."""

    backend: str = "none"
    resident_ratio: float | None = None
    resident_gib: float | None = None
    storage_path: Path | None = None
    io_depth: int = 2
    prefetch_layer_horizon: int = 1
    prefetch_candidates: int = 0
    max_inflight_gib: float = 0.25

    def __post_init__(self) -> None:
        if self.backend not in {"none", "memory", "nvme"}:
            raise ValueError(f"unsupported expert offload backend: {self.backend}")
        if self.resident_ratio is not None and self.resident_gib is not None:
            raise ValueError("resident_ratio and resident_gib are mutually exclusive")
        if self.resident_ratio is not None and not 0 < self.resident_ratio <= 1:
            raise ValueError("resident_ratio must be in (0, 1]")
        if self.resident_gib is not None and self.resident_gib <= 0:
            raise ValueError("resident_gib must be positive")
        if self.io_depth <= 0:
            raise ValueError("io_depth must be positive")
        if self.prefetch_layer_horizon <= 0:
            raise ValueError("prefetch_layer_horizon must be positive")
        if self.prefetch_candidates < 0:
            raise ValueError("prefetch_candidates cannot be negative")
        if self.max_inflight_gib <= 0:
            raise ValueError("max_inflight_gib must be positive")
        if self.backend == "none":
            return
        if self.resident_ratio is None and self.resident_gib is None:
            raise ValueError(
                "an enabled expert offload backend needs a resident budget"
            )
        if self.backend == "nvme" and self.storage_path is None:
            raise ValueError("the nvme backend requires storage_path")
