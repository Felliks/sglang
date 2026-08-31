import os
from pathlib import Path

import pytest
import torch
from sglang.srt.mem_cache.active_sparse_kv import (
    ActiveSparseKVConfig,
    ActiveSparseQSAKVCoordinator,
)


class _MockFullPool:
    def __init__(self, token_capacity: int):
        shape = (token_capacity, 2, 256)
        self.k = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda")]
        self.v = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda")]

    def get_key_buffer(self, layer: int):
        return self.k[layer]

    def get_value_buffer(self, layer: int):
        return self.v[layer]


class _MockHybridPool:
    uses_bounded_full_kv = True
    qsa_compress_ratio = 4
    full_layer_nums = 1
    full_kv_token_capacity = 8
    logical_size = 32
    page_size = 4
    full_attention_layer_id_mapping = {0: 0}

    def __init__(self):
        self.full_kv_pool = _MockFullPool(self.full_kv_token_capacity)
        self.active_kv_coordinator = None

    def _transfer_full_attention_id(self, layer: int) -> int:
        return self.full_attention_layer_id_mapping[layer]

    def get_key_buffer(self, layer: int):
        return self.full_kv_pool.get_key_buffer(
            self._transfer_full_attention_id(layer)
        )

    def get_value_buffer(self, layer: int):
        return self.full_kv_pool.get_value_buffer(
            self._transfer_full_attention_id(layer)
        )

    def register_active_kv_coordinator(self, coordinator) -> None:
        self.active_kv_coordinator = coordinator


class _MockAllocator:
    def __init__(self, pool):
        self.pool = pool

    def get_kvcache(self):
        return self.pool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_nvme_write_evict_read_roundtrip(tmp_path: Path) -> None:
    if not os.environ.get("SGLANG_ACTIVE_KV_IO_URING_LIBRARY"):
        pytest.skip("active-KV io_uring helper is not configured")
    pool = _MockHybridPool()
    config = ActiveSparseKVConfig(
        backend="nvme",
        path=tmp_path,
        max_bytes=1024 * 1024,
        io_depth=4,
    )
    coordinator = ActiveSparseQSAKVCoordinator(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=_MockAllocator(pool),
        config=config,
        device="cuda",
        rank=0,
        max_running_requests=1,
    )
    assert coordinator.io.staging.data_ptr() % 4096 == 0
    assert coordinator.io.staging.stride() == (coordinator.record_bytes, 1)
    try:
        logical = torch.arange(4, 8, dtype=torch.int64, device="cuda")
        plan = coordinator.prepare_write(0, logical)
        expected_k = torch.arange(
            4 * 2 * 256, dtype=torch.float32, device="cuda"
        ).reshape(4, 2, 256).to(torch.bfloat16)
        expected_v = -expected_k
        pool.full_kv_pool.k[0][plan.physical_locs] = expected_k
        pool.full_kv_pool.v[0][plan.physical_locs] = expected_v
        coordinator.commit_write(plan)

        # Fill the only other hot block and then a third one, forcing block 1
        # out before the exact readback.
        for start in (8, 12):
            other = torch.arange(start, start + 4, device="cuda")
            other_plan = coordinator.prepare_write(0, other)
            pool.full_kv_pool.k[0][other_plan.physical_locs].fill_(start)
            pool.full_kv_pool.v[0][other_plan.physical_locs].fill_(-start)
            coordinator.commit_write(other_plan)

        assert coordinator.directory.lookup(0, 1) == -1
        restored = coordinator.materialize_slots(0, logical)
        torch.testing.assert_close(pool.full_kv_pool.k[0][restored], expected_k)
        torch.testing.assert_close(pool.full_kv_pool.v[0][restored], expected_v)
        assert coordinator.cache_metrics()["read_misses"] == 1
    finally:
        extent_path = coordinator.extent.path
        coordinator.destroy()
        assert not extent_path.exists()
