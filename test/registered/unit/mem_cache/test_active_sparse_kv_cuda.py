import os
from pathlib import Path

import pytest
import torch
from sglang.srt.mem_cache.active_sparse_kv import (
    ActiveSparseKVConfig,
    ActiveSparseQSAKVCoordinator,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
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
        page_cache_bytes=8192,
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
        resident = coordinator.resolve_resident_slots(0, logical)
        assert resident is not None
        torch.testing.assert_close(resident, plan.physical_locs.to(torch.int32))

        # Fill the only other hot block and then a third one, forcing block 1
        # out before the exact readback.
        for start in (8, 12):
            other = torch.arange(start, start + 4, device="cuda")
            other_plan = coordinator.prepare_write(0, other)
            pool.full_kv_pool.k[0][other_plan.physical_locs].fill_(start)
            pool.full_kv_pool.v[0][other_plan.physical_locs].fill_(-start)
            coordinator.commit_write(other_plan)

        assert coordinator.directory.lookup(0, 1) == -1
        assert coordinator.resolve_resident_slots(0, logical) is None
        restored = coordinator.materialize_slots(0, logical)
        torch.testing.assert_close(pool.full_kv_pool.k[0][restored], expected_k)
        torch.testing.assert_close(pool.full_kv_pool.v[0][restored], expected_v)
        resident = coordinator.resolve_resident_slots(0, logical)
        assert resident is not None
        torch.testing.assert_close(resident, restored)
        assert coordinator.cache_metrics()["read_misses"] == 1

        # Evict the restored block once more.  The bounded dual-path cache now
        # serves its exact record through the buffered/page-cache fd while
        # never changing the BF16 payload.
        for start in (16, 20):
            other = torch.arange(start, start + 4, device="cuda")
            other_plan = coordinator.prepare_write(0, other)
            pool.full_kv_pool.k[0][other_plan.physical_locs].fill_(start)
            pool.full_kv_pool.v[0][other_plan.physical_locs].fill_(-start)
            coordinator.commit_write(other_plan)
        restored_again = coordinator.materialize_slots(0, logical)
        torch.testing.assert_close(
            pool.full_kv_pool.k[0][restored_again], expected_k
        )
        torch.testing.assert_close(
            pool.full_kv_pool.v[0][restored_again], expected_v
        )
        metrics = coordinator.cache_metrics()
        assert metrics["direct_reads"] == 1
        assert metrics["page_cache_reads"] == 1
    finally:
        extent_path = coordinator.extent.path
        coordinator.destroy()
        assert not extent_path.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_prefill_accepts_absolute_hot_slots() -> None:
    torch.manual_seed(7)
    rows, q_heads, kv_heads, dim = 4, 8, 2, 256
    q = torch.randn(rows, q_heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(32, kv_heads, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    slots = torch.full((rows, 2048), -1, dtype=torch.int32, device="cuda")
    for row in range(rows):
        visible = 9 + row
        slots[row, :visible] = torch.randperm(32, device="cuda")[:visible]
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([12], dtype=torch.int32, device="cuda")
    scale = dim**-0.5
    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, slots, cu_q, cu_k, kv_lens, scale
    )
    expected = qsa_sparse_attention_reference(q, k, v, slots, scale)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
