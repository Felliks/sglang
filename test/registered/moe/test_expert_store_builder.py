from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertTensors,
)
from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.storage.builder import (
    ExpertStoreWriter,
    build_marlin_expert_store,
)
from sglang.srt.layers.moe.expert_offload.storage.manifest import (
    ExpertStoreManifest,
    TensorSegment,
)
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    DirectExpertStorage,
)


def _bundle(offset: int) -> MarlinExpertTensors:
    values = torch.arange(8, dtype=torch.uint8).reshape(2, 4) + offset
    scales = torch.arange(4, dtype=torch.float32).reshape(2, 2) + offset
    return MarlinExpertTensors(
        w13_qweight=values,
        w2_qweight=values + 10,
        w13_scales=scales,
        w2_scales=scales + 10,
        w13_global_scale=scales[:, :1] + 20,
        w2_global_scale=scales[:, :1] + 30,
    )


def test_marlin_store_round_trip_is_atomic_and_checksummed(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    completed = build_marlin_expert_store(
        manifest_path,
        [_bundle(0), _bundle(40)],
        model_fingerprint="test-model",
        alignment=4096,
        direct=False,
    )

    assert completed.num_records == 4
    assert len(completed.record_sha256) == 4
    assert manifest_path.exists()
    assert not (tmp_path / "experts.marlin.bin.partial").exists()
    with DirectExpertStorage(
        manifest_path, direct=False, verify_checksums=True
    ) as storage:
        record = storage.submit(ExpertIdentity(1, 1)).result()
        with record:
            first = completed.tensor_segments[0]
            assert bytes(record.view()[first.offset : first.offset + 4]) == bytes(
                [44, 45, 46, 47]
            )


def test_incomplete_writer_publishes_no_final_store(tmp_path) -> None:
    manifest = ExpertStoreManifest(
        data_file="experts.bin",
        alignment=4096,
        record_bytes=4096,
        num_layers=1,
        experts_per_layer=2,
        tensor_segments=(TensorSegment("weight", 0, 4, "uint8", (4,)),),
        model_fingerprint="model",
        packing_fingerprint="packing",
    )
    manifest_path = tmp_path / "manifest.json"
    with ExpertStoreWriter(manifest_path, manifest, direct=False) as writer:
        writer.write_record(
            lambda destination: destination.__setitem__(slice(0, 4), b"abcd")
        )
        with pytest.raises(RuntimeError, match="1 of 2"):
            writer.finish()

    assert not manifest_path.exists()
    assert not (tmp_path / "experts.bin").exists()
    assert (tmp_path / "experts.bin.partial").exists()
