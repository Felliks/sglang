from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import torch
from sglang.kernels.ops.kvcache.active_sparse_kv import (
    pack_qsa_records,
    unpack_qsa_records,
)
from sglang.srt.mem_cache.active_sparse_kv.config import ActiveSparseKVConfig
from sglang.srt.mem_cache.active_sparse_kv.directory import ActiveKVBlockDirectory
from sglang.srt.mem_cache.active_sparse_kv.layout import ActiveSparseKVLayout
from sglang.srt.mem_cache.active_sparse_kv.storage import (
    ActiveKVExtent,
    NativeActiveKVUring,
)

logger = logging.getLogger(__name__)


class ActiveSparseKVTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


@dataclass(frozen=True)
class ActiveKVWritePlan:
    layer_offset: int
    physical_locs: torch.Tensor
    completed: tuple[tuple[int, int], ...]


class ActiveSparseQSAKVCoordinator:
    """Exact compressed-QSA KV tier backed by a process-owned NVMe extent.

    Full BF16 K/V records are authoritative on NVMe.  The regular MHA pool is
    reinterpreted as a bounded hot block cache; the trained QSA index remains
    fully resident and selects the records that must be materialized.  All
    storage errors fail the request rather than silently falling back to stale
    slots or lossy KV.

    This first backend deliberately runs without CUDA graphs.  Its blocking
    boundaries are explicit so the subsequent fused planner/overlap work can
    replace them without changing ownership or correctness semantics.
    """

    requires_staging = False

    def __init__(
        self,
        *,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        config: ActiveSparseKVConfig,
        device: str,
        rank: int,
        max_running_requests: int,
    ) -> None:
        if config.backend != "nvme":
            raise ValueError("ActiveSparseQSAKVCoordinator requires NVMe config")
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.pool = token_to_kv_pool_allocator.get_kvcache()
        if not getattr(self.pool, "uses_bounded_full_kv", False):
            raise RuntimeError("active QSA coordinator requires a bounded full-KV pool")
        self.device = torch.device(device)
        if max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")
        self.max_running_requests = int(max_running_requests)
        self.block_tokens = int(self.pool.qsa_compress_ratio)
        self.num_layers = int(self.pool.full_layer_nums)
        self.hot_blocks = int(self.pool.full_kv_token_capacity // self.block_tokens)
        first_layer_id = next(iter(self.pool.full_attention_layer_id_mapping))
        k0 = self.pool.get_key_buffer(first_layer_id)
        v0 = self.pool.get_value_buffer(first_layer_id)
        token_bytes = int(k0.stride(0) * k0.element_size()) + int(
            v0.stride(0) * v0.element_size()
        )
        self.record_bytes = token_bytes * self.block_tokens
        self.layout = ActiveSparseKVLayout(
            num_layers=self.num_layers,
            logical_token_capacity=int(self.pool.logical_size),
            page_size=int(self.pool.page_size),
            block_tokens=self.block_tokens,
            record_bytes=self.record_bytes,
        )
        self.extent = ActiveKVExtent(config, self.layout, rank=rank)
        try:
            self.io = NativeActiveKVUring(
                self.extent.path,
                record_bytes=self.record_bytes,
                io_depth=config.io_depth,
            )
        except BaseException:
            self.extent.close()
            raise
        self.directory = ActiveKVBlockDirectory(
            num_layers=self.num_layers,
            logical_blocks=self.layout.block_capacity,
            hot_blocks=self.hot_blocks,
        )
        self.pool.register_active_kv_coordinator(self)
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._ready_reqs = []
        self._closed = False
        self._reads = 0
        self._read_misses = 0
        self._writes = 0
        self._materialize_calls = 0
        self._log_interval = config.log_interval
        logger.warning(
            "Exact active QSA KV enabled: extent=%s bytes=%d hot_blocks=%d "
            "record_bytes=%d",
            self.extent.path,
            self.layout.file_bytes,
            self.hot_blocks,
            self.record_bytes,
        )

    @property
    def materialization_block_capacity(self) -> int:
        # At most one incomplete compression group is pinned per live request.
        # The physical pool sizing already adds that reserve, so the configured
        # QSA hot budget is the remainder.
        return max(
            1,
            self.hot_blocks - self.max_running_requests,
        )

    def _local_layer(self, layer_id: int) -> int:
        return int(self.pool._transfer_full_attention_id(int(layer_id)))

    def prepare_write(
        self, layer_id: int, logical_locs: torch.Tensor
    ) -> ActiveKVWritePlan:
        layer = self._local_layer(layer_id)
        logical_cpu = logical_locs.detach().to("cpu", dtype=torch.int64).flatten()
        block_to_offsets: dict[int, set[int]] = {}
        for logical in logical_cpu.tolist():
            block = self.layout.block_slot(int(logical))
            block_to_offsets.setdefault(block, set()).add(
                int(logical) % self.block_tokens
            )
        hot_by_block = {}
        completed = []
        for block, offsets in block_to_offsets.items():
            hot = self.directory.begin_write(
                layer, block, starts_block=0 in offsets
            )
            hot_by_block[block] = hot
            if self.block_tokens - 1 in offsets:
                completed.append((block, hot))
        physical = [
            hot_by_block[int(logical) // self.block_tokens] * self.block_tokens
            + int(logical) % self.block_tokens
            for logical in logical_cpu.tolist()
        ]
        return ActiveKVWritePlan(
            layer_offset=layer,
            physical_locs=torch.tensor(
                physical, dtype=torch.int64, device=logical_locs.device
            ).reshape(logical_locs.shape),
            completed=tuple(completed),
        )

    def commit_write(self, plan: ActiveKVWritePlan) -> None:
        if not plan.completed:
            return
        k_buffer = self.pool.full_kv_pool.get_key_buffer(plan.layer_offset)
        v_buffer = self.pool.full_kv_pool.get_value_buffer(plan.layer_offset)
        for start in range(0, len(plan.completed), self.io.io_depth):
            batch = plan.completed[start : start + self.io.io_depth]
            logical_blocks = [item[0] for item in batch]
            hot_blocks = torch.tensor(
                [item[1] for item in batch], dtype=torch.int32, device=self.device
            )
            staging = self.io.staging[: len(batch)]
            pack_qsa_records(
                source_k=k_buffer,
                source_v=v_buffer,
                source_blocks=hot_blocks,
                staging=staging,
                num_records=len(batch),
                block_tokens=self.block_tokens,
            )
            torch.cuda.current_stream(self.device).synchronize()
            offsets = [
                self.layout.record_offset(plan.layer_offset, block)
                for block in logical_blocks
            ]
            self.io.write(offsets, staging)
            for block in logical_blocks:
                self.directory.finish_write(plan.layer_offset, block)
            self._writes += len(batch)

    def materialize_slots(
        self, layer_id: int, logical_slots: torch.Tensor
    ) -> torch.Tensor:
        """Return hot physical token slots for global logical KV slots."""

        layer = self._local_layer(layer_id)
        shape = logical_slots.shape
        logical_cpu = logical_slots.detach().to("cpu", dtype=torch.int64).flatten()
        valid_slots = [int(slot) for slot in logical_cpu.tolist() if int(slot) >= 0]
        blocks = tuple(dict.fromkeys(slot // self.block_tokens for slot in valid_slots))
        placement = self.directory.place(
            layer, blocks, require_authoritative=True
        )
        self._reads += len(blocks)
        self._read_misses += len(placement.misses)
        self._materialize_calls += 1
        for start in range(0, len(placement.misses), self.io.io_depth):
            batch = placement.misses[start : start + self.io.io_depth]
            offsets = [
                self.layout.record_offset(layer, logical_block)
                for logical_block, _ in batch
            ]
            staging = self.io.read(offsets)
            destination = torch.tensor(
                [hot_block for _, hot_block in batch],
                dtype=torch.int32,
                device=self.device,
            )
            unpack_qsa_records(
                staging=staging,
                destination_k=self.pool.full_kv_pool.get_key_buffer(layer),
                destination_v=self.pool.full_kv_pool.get_value_buffer(layer),
                destination_blocks=destination,
                num_records=len(batch),
                block_tokens=self.block_tokens,
            )
            # The next O_DIRECT batch reuses the fixed staging rows.  Do not let
            # the drive overwrite a row before CUDA has consumed it.
            torch.cuda.current_stream(self.device).synchronize()
        if self._materialize_calls % self._log_interval == 0:
            logger.info(
                "Exact active QSA KV metrics after %d materializations: %s",
                self._materialize_calls,
                self.cache_metrics(),
            )
        translated = []
        for slot in logical_cpu.tolist():
            if slot < 0:
                translated.append(-1)
                continue
            block = int(slot) // self.block_tokens
            hot = self.directory.lookup(layer, block)
            if hot < 0:
                raise RuntimeError("active-KV block lost placement after materialize")
            translated.append(hot * self.block_tokens + int(slot) % self.block_tokens)
        return torch.tensor(
            translated, dtype=torch.int32, device=logical_slots.device
        ).reshape(shape)

    # Scheduler lifecycle compatibility.  Unlike host HiSparse, prefill has
    # already written through and there is no post-prefill staging DMA.
    def admit_request_into_staging(self, req) -> None:
        req.hisparse_staging = False

    def admit_request_direct(self, req) -> None:
        self.admit_request_into_staging(req)

    def collect_ready_reqs(self):
        ready, self._ready_reqs = self._ready_reqs, []
        return ready

    def has_ongoing_staging(self) -> bool:
        return False

    def map_last_loc_to_buffer(self, *args, **kwargs) -> None:
        return None

    def wait_for_pending_backup(self) -> None:
        return None

    def set_decode_producer_stream(self, stream) -> None:
        return None

    def request_finished(self, req) -> None:
        return None

    def retract_req(self, req) -> None:
        return None

    def get_token_stats(self) -> ActiveSparseKVTokenStats:
        hot = sum(
            1
            for layer in self.directory._hot_to_logical
            for logical in layer
            if logical >= 0
        )
        authoritative = sum(
            1
            for layer in self.directory._authoritative
            for present in layer
            if present
        )
        hot_capacity = self.num_layers * self.hot_blocks
        host_capacity = self.num_layers * self.layout.block_capacity
        return ActiveSparseKVTokenStats(
            device_tokens=hot * self.block_tokens,
            device_token_usage=hot / hot_capacity,
            host_tokens=authoritative * self.block_tokens,
            host_token_usage=authoritative / host_capacity,
        )

    def cache_metrics(self) -> dict[str, float | int]:
        return {
            "selected_blocks": self._reads,
            "read_misses": self._read_misses,
            "hit_rate": 1.0 - self._read_misses / max(self._reads, 1),
            "written_blocks": self._writes,
        }

    def destroy(self) -> None:
        if self._closed:
            return
        logger.warning("Exact active QSA KV closing: metrics=%s", self.cache_metrics())
        self.io.close()
        self.extent.close()
        self._closed = True
