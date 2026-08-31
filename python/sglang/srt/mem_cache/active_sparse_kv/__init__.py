"""Exact active sparse-KV storage tiers.

This package contains storage and layout primitives used by native sparse
attention backends.  It is deliberately separate from HiCache: HiCache moves
inactive prefix pages, whereas an active sparse-KV tier materializes the
selected working set on every decode step.
"""

from sglang.srt.mem_cache.active_sparse_kv.config import (
    ActiveSparseKVConfig,
    active_qsa_hot_token_capacity,
)
from sglang.srt.mem_cache.active_sparse_kv.coordinator import (
    ActiveSparseQSAKVCoordinator,
)
from sglang.srt.mem_cache.active_sparse_kv.directory import (
    ActiveKVBlockDirectory,
    BlockPlacement,
)
from sglang.srt.mem_cache.active_sparse_kv.layout import ActiveSparseKVLayout
from sglang.srt.mem_cache.active_sparse_kv.storage import (
    ActiveKVExtent,
    NativeActiveKVUring,
)

__all__ = [
    "ActiveKVExtent",
    "ActiveKVBlockDirectory",
    "ActiveSparseKVConfig",
    "ActiveSparseKVLayout",
    "ActiveSparseQSAKVCoordinator",
    "NativeActiveKVUring",
    "BlockPlacement",
    "active_qsa_hot_token_capacity",
]
