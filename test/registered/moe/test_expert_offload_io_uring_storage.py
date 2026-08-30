from __future__ import annotations

import ctypes
import hashlib
import os
from collections import deque

import pytest
from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.storage.manifest import (
    ExpertStoreManifest,
    TensorSegment,
)
from sglang.srt.layers.moe.expert_offload.storage.nvme_uring import (
    IoUringExpertStorage,
)


class _FakeNativeRing:
    def __init__(self, library_path, data_path, buffers) -> None:
        del library_path
        self.fd = os.open(data_path, os.O_RDONLY)
        self.buffers = buffers
        self.completions = deque()

    def submit(self, index, buffer, offset) -> None:
        payload = os.pread(self.fd, buffer.size, offset)
        ctypes.memmove(buffer.address, payload, len(payload))
        self.completions.append((index, len(payload)))

    def wait(self):
        return self.completions.popleft()

    def close(self) -> None:
        os.close(self.fd)


def _store(tmp_path, *, corrupt_checksum: bool = False):
    alignment = 4096
    records = [bytes([index]) * alignment for index in range(4)]
    data_path = tmp_path / "experts.bin"
    data_path.write_bytes(b"".join(records))
    checksums = [hashlib.sha256(record).hexdigest() for record in records]
    if corrupt_checksum:
        checksums[2] = "0" * 64
    manifest = ExpertStoreManifest(
        data_file=data_path.name,
        alignment=alignment,
        record_bytes=alignment,
        num_layers=2,
        experts_per_layer=2,
        tensor_segments=(
            TensorSegment(
                name="weight", offset=0, nbytes=16, dtype="uint8", shape=(16,)
            ),
        ),
        model_fingerprint="model-sha256",
        packing_fingerprint="marlin-v1",
        record_sha256=tuple(checksums),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def test_io_uring_storage_preserves_identity_and_buffer_leases(tmp_path) -> None:
    storage = IoUringExpertStorage(
        _store(tmp_path),
        io_depth=2,
        verify_checksums=True,
        _native_factory=_FakeNativeRing,
    )
    first = storage.submit(ExpertIdentity(1, 0)).result(timeout=2)
    second = storage.submit(ExpertIdentity(0, 1)).result(timeout=2)
    assert first.identity == ExpertIdentity(1, 0)
    assert first.view()[0] == 2
    assert second.view()[0] == 1
    with pytest.raises(RuntimeError, match="leased"):
        storage.close()
    first.release()
    second.release()
    storage.close()


def test_io_uring_storage_fails_closed_on_checksum_mismatch(tmp_path) -> None:
    with IoUringExpertStorage(
        _store(tmp_path, corrupt_checksum=True),
        io_depth=1,
        verify_checksums=True,
        _native_factory=_FakeNativeRing,
    ) as storage:
        future = storage.submit(ExpertIdentity(1, 0))
        with pytest.raises(OSError, match="checksum mismatch"):
            future.result(timeout=2)
