from __future__ import annotations

import hashlib

import pytest
from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.storage.manifest import (
    ExpertStoreManifest,
    TensorSegment,
)
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    DirectExpertStorage,
)


def _store(tmp_path, *, alignment: int = 4096):
    records = [bytes([index]) * alignment for index in range(4)]
    data_path = tmp_path / "experts.bin"
    data_path.write_bytes(b"".join(records))
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
        record_sha256=tuple(hashlib.sha256(record).hexdigest() for record in records),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def test_manifest_round_trip_and_record_geometry(tmp_path) -> None:
    manifest_path = _store(tmp_path)
    manifest = ExpertStoreManifest.load(manifest_path)
    identity = ExpertIdentity(1, 1)
    assert manifest.record_index(identity) == 3
    assert manifest.record_offset(identity) == 3 * 4096
    assert manifest.file_bytes == 4 * 4096


def test_bounded_async_reader_and_checksum(tmp_path) -> None:
    manifest_path = _store(tmp_path)
    with DirectExpertStorage(
        manifest_path, io_depth=2, direct=False, verify_checksums=True
    ) as storage:
        first = storage.submit(ExpertIdentity(1, 0))
        second = storage.submit(ExpertIdentity(0, 1))
        with first.result() as record:
            assert record.view()[0] == 2
        with second.result() as record:
            assert record.view()[0] == 1


def test_manifest_and_file_mismatches_fail_closed(tmp_path) -> None:
    manifest_path = _store(tmp_path)
    (tmp_path / "experts.bin").write_bytes(b"short")
    with pytest.raises(ValueError, match="expected"):
        DirectExpertStorage(manifest_path, direct=False)

    with pytest.raises(ValueError, match="outside"):
        ExpertStoreManifest.load(manifest_path).record_index(ExpertIdentity(2, 0))


def test_storage_refuses_to_close_with_a_leased_buffer(tmp_path) -> None:
    storage = DirectExpertStorage(_store(tmp_path), io_depth=1, direct=False)
    completed = storage.submit(ExpertIdentity(0, 0)).result()
    with pytest.raises(RuntimeError, match="leased"):
        storage.close()
    completed.release()
    storage.close()
