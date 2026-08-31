import fcntl
import os
from pathlib import Path

import pytest
from sglang.srt.mem_cache.active_sparse_kv import (
    ActiveKVBlockDirectory,
    ActiveKVExtent,
    ActiveSparseKVConfig,
    ActiveSparseKVLayout,
    active_qsa_hot_token_capacity,
)


def test_nvme_config_is_explicit_and_bounded() -> None:
    config = ActiveSparseKVConfig.from_hisparse_extra_config(
        {
            "active_kv_backend": "nvme",
            "active_kv_path": "/var/tmp/active-kv",
            "active_kv_max_bytes": 20 * 1024**3,
            "active_kv_min_free_bytes": 80 * 1024**3,
            "active_kv_io_depth": 64,
            "active_kv_page_cache_bytes": 512 * 1024**2,
        }
    )
    assert config.path == Path("/var/tmp/active-kv")
    assert config.max_bytes == 20 * 1024**3
    assert config.io_depth == 64
    assert config.page_cache_bytes == 512 * 1024**2


@pytest.mark.parametrize(
    "extra",
    [
        {"active_kv_backend": "nvme", "active_kv_max_bytes": 1},
        {"active_kv_backend": "nvme", "active_kv_path": "/tmp/x"},
        {"active_kv_backend": "unknown"},
        {"active_kv_backend": "host", "active_kv_path": "/tmp/x"},
        {"active_kv_backend": "host", "active_kv_page_cache_bytes": 4096},
        {
            "active_kv_backend": "nvme",
            "active_kv_path": "/tmp/x",
            "active_kv_max_bytes": 1,
            "active_kv_io_depth": 0,
        },
        {
            "active_kv_backend": "nvme",
            "active_kv_path": "/tmp/x",
            "active_kv_max_bytes": 4096,
            "active_kv_page_cache_bytes": 8192,
        },
    ],
)
def test_invalid_active_kv_config_fails_closed(extra) -> None:
    with pytest.raises(ValueError):
        ActiveSparseKVConfig.from_hisparse_extra_config(extra)


def test_qwen_compressed_qsa_layout() -> None:
    layout = ActiveSparseKVLayout(
        num_layers=12,
        logical_token_capacity=262_144,
        page_size=64,
        block_tokens=4,
        record_bytes=8192,
    )
    assert layout.block_capacity == 65_552
    # 262144 target tokens consume exactly 6 GiB; the allocator's reserved
    # 64-token page adds 1.5 MiB to the fixed extent.
    assert layout.file_bytes == 6_444_023_808
    assert layout.block_slot(0) == 0
    assert layout.block_slot(3) == 0
    assert layout.block_slot(4) == 1
    assert layout.record_offset(1, 0) == layout.layer_bytes
    layout.validate_capacity(20 * 1024**3)


def test_hot_pool_capacity_includes_one_pending_block_per_request() -> None:
    assert (
        active_qsa_hot_token_capacity(
            device_buffer_tokens=8192,
            max_running_requests=2,
            block_tokens=4,
            page_size=64,
        )
        == 16_448
    )


def test_layout_rejects_unaligned_or_oversized_extents() -> None:
    with pytest.raises(ValueError, match="aligned"):
        ActiveSparseKVLayout(12, 262_144, 64, 4, 6144)
    layout = ActiveSparseKVLayout(12, 262_144, 64, 4, 8192)
    with pytest.raises(ValueError, match="exceeds"):
        layout.validate_capacity(1024)


def test_block_directory_keeps_partial_writes_pinned_and_reads_exact() -> None:
    directory = ActiveKVBlockDirectory(num_layers=1, logical_blocks=8, hot_blocks=2)
    first = directory.begin_write(0, 3, starts_block=True)
    assert directory.lookup(0, 3) == first
    # The hot partial block is readable for QSA's current-tail tokens, but an
    # unwritten cold block must never be fabricated from NVMe.
    assert directory.place(0, [3], require_authoritative=True).misses == ()
    with pytest.raises(RuntimeError, match="before authoritative write"):
        directory.place(0, [6], require_authoritative=True)

    directory.finish_write(0, 3)
    assert directory.place(0, [3], require_authoritative=True).misses == ()
    directory.begin_write(0, 4, starts_block=True)
    directory.finish_write(0, 4)
    placement = directory.place(0, [5], require_authoritative=False)
    assert len(placement.misses) == 1
    assert directory.lookup(0, 3) == -1


def test_block_directory_refuses_to_evict_only_partial_blocks() -> None:
    directory = ActiveKVBlockDirectory(num_layers=1, logical_blocks=8, hot_blocks=2)
    directory.begin_write(0, 0, starts_block=True)
    directory.begin_write(0, 1, starts_block=True)
    with pytest.raises(RuntimeError, match="exhausted by pinned"):
        directory.begin_write(0, 2, starts_block=True)


def test_block_directory_protects_later_hits_before_allocating_misses() -> None:
    directory = ActiveKVBlockDirectory(num_layers=1, logical_blocks=8, hot_blocks=2)
    directory.place(0, [0, 1], require_authoritative=False)
    placement = directory.place(0, [2, 0], require_authoritative=False)
    assert len(placement.misses) == 1
    assert directory.lookup(0, 0) >= 0
    assert directory.lookup(0, 1) == -1


def test_extent_reclaims_only_unlocked_crash_artifacts(tmp_path: Path) -> None:
    stale = tmp_path / "active-kv-rank9-pid999999.bin"
    stale.write_bytes(b"stale")
    live = tmp_path / "active-kv-rank8-pid888888.bin"
    live_fd = os.open(live, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    fcntl.flock(live_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = ActiveSparseKVConfig(
        backend="nvme", path=tmp_path, max_bytes=1024 * 1024
    )
    layout = ActiveSparseKVLayout(1, 4, 4, 4, 4096)
    try:
        extent = ActiveKVExtent(config, layout, rank=0)
        assert not stale.exists()
        assert live.exists()
        extent.close()
    finally:
        os.close(live_fd)
