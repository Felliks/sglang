from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ActiveSparseKVConfig:
    """Fail-closed configuration for an authoritative active-KV store."""

    backend: str = "host"
    path: Path | None = None
    max_bytes: int = 0
    min_free_bytes: int = 0
    io_depth: int = 64
    log_interval: int = 4096
    verify_reads: bool = False

    @classmethod
    def from_hisparse_extra_config(
        cls, extra_config: Mapping[str, Any]
    ) -> "ActiveSparseKVConfig":
        backend = str(extra_config.get("active_kv_backend", "host")).lower()
        if backend not in {"host", "nvme"}:
            raise ValueError(
                "active_kv_backend must be one of {'host', 'nvme'}, "
                f"got {backend!r}"
            )
        raw_path = extra_config.get("active_kv_path")
        path = Path(raw_path).expanduser() if raw_path is not None else None
        max_bytes = _non_negative_int(
            extra_config.get("active_kv_max_bytes", 0), "active_kv_max_bytes"
        )
        min_free_bytes = _non_negative_int(
            extra_config.get("active_kv_min_free_bytes", 0),
            "active_kv_min_free_bytes",
        )
        io_depth = _positive_int(
            extra_config.get("active_kv_io_depth", 64), "active_kv_io_depth"
        )
        if io_depth > 4096:
            raise ValueError("active_kv_io_depth must not exceed 4096")
        log_interval = _positive_int(
            extra_config.get("active_kv_log_interval", 4096),
            "active_kv_log_interval",
        )
        verify_reads = extra_config.get("active_kv_verify_reads", False)
        if not isinstance(verify_reads, bool):
            raise ValueError("active_kv_verify_reads must be a boolean")
        if backend == "nvme":
            if path is None:
                raise ValueError("active_kv_path is required for the NVMe backend")
            if max_bytes == 0:
                raise ValueError(
                    "active_kv_max_bytes is required for the NVMe backend"
                )
        elif path is not None or max_bytes or min_free_bytes:
            raise ValueError(
                "active_kv_path/active_kv_max_bytes/active_kv_min_free_bytes "
                "are only valid with active_kv_backend='nvme'"
            )
        return cls(
            backend=backend,
            path=path,
            max_bytes=max_bytes,
            min_free_bytes=min_free_bytes,
            io_depth=io_depth,
            log_interval=log_interval,
            verify_reads=verify_reads,
        )


def active_qsa_hot_token_capacity(
    *,
    device_buffer_tokens: int,
    max_running_requests: int,
    block_tokens: int,
    page_size: int,
) -> int:
    """Aggregate physical token slots for per-request LRU + newest block."""

    for value, name in (
        (device_buffer_tokens, "device_buffer_tokens"),
        (max_running_requests, "max_running_requests"),
        (block_tokens, "block_tokens"),
        (page_size, "page_size"),
    ):
        _positive_int(value, name)
    if device_buffer_tokens % block_tokens:
        raise ValueError("device_buffer_tokens must be divisible by block_tokens")
    if page_size % block_tokens:
        raise ValueError("page_size must be divisible by block_tokens")
    requested = (device_buffer_tokens + block_tokens) * max_running_requests
    return (requested + page_size - 1) // page_size * page_size


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value
