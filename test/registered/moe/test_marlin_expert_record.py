from __future__ import annotations

import ctypes
from dataclasses import replace

import pytest
import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertTensors,
)
from sglang.srt.layers.moe.expert_offload.adapters.marlin_record import (
    MarlinRecordPublisher,
    marlin_packing_fingerprint,
    marlin_tensor_segments,
)
from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    AlignedHostBuffer,
    DirectRead,
)


def _bundle(num_experts: int) -> MarlinExpertTensors:
    values = torch.arange(num_experts * 4, dtype=torch.uint8).reshape(num_experts, 4)
    scales = torch.arange(num_experts * 2, dtype=torch.float32).reshape(num_experts, 2)
    return MarlinExpertTensors(
        w13_qweight=values,
        w2_qweight=values + 20,
        w13_scales=scales,
        w2_scales=scales + 20,
        w13_global_scale=scales[:, :1] + 40,
        w2_global_scale=scales[:, :1] + 60,
    )


def test_layout_and_cpu_publication_are_byte_exact() -> None:
    source = _bundle(2)
    segments = marlin_tensor_segments(source)
    assert marlin_packing_fingerprint(segments) == marlin_packing_fingerprint(segments)
    record_bytes = sum(segment.nbytes for segment in segments)
    buffer = AlignedHostBuffer(record_bytes, 8)
    for segment in segments:
        row = getattr(source, segment.name)[1]
        payload = bytes(row.contiguous().reshape(-1).view(torch.uint8).tolist())
        ctypes.memmove(buffer.address + segment.offset, payload, len(payload))

    released = []
    completed = DirectRead(ExpertIdentity(0, 1), buffer, released.append)
    destination = MarlinExpertTensors(
        **{
            name: torch.zeros_like(getattr(source, name))
            for name in (
                "w13_qweight",
                "w2_qweight",
                "w13_scales",
                "w2_scales",
                "w13_global_scale",
                "w2_global_scale",
            )
        }
    )
    publication = MarlinRecordPublisher(segments).publish(
        completed, destination, slot_id=0
    )

    assert publication.ready()
    assert released == [buffer]
    for segment in segments:
        assert torch.equal(
            getattr(destination, segment.name)[0], getattr(source, segment.name)[1]
        )
    buffer.close()


def test_publication_rejects_layout_mismatch() -> None:
    source = _bundle(2)
    segments = marlin_tensor_segments(source)
    buffer = AlignedHostBuffer(sum(segment.nbytes for segment in segments), 8)
    completed = DirectRead(ExpertIdentity(0, 0), buffer, lambda _: None)
    destination = replace(
        _bundle(2), w13_qweight=torch.zeros((2, 3), dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        MarlinRecordPublisher(segments).publish(completed, destination, 0)
    completed.release()
    buffer.close()


def test_layout_accepts_per_expert_scalar_global_scales() -> None:
    source = _bundle(2)
    scalar_scales = MarlinExpertTensors(
        w13_qweight=source.w13_qweight,
        w2_qweight=source.w2_qweight,
        w13_scales=source.w13_scales,
        w2_scales=source.w2_scales,
        w13_global_scale=torch.tensor([1.0, 2.0]),
        w2_global_scale=torch.tensor([3.0, 4.0]),
    )

    segments = marlin_tensor_segments(scalar_scales)
    assert segments[-1].shape == ()
    assert segments[-1].nbytes == 4

    record_bytes = sum(segment.nbytes for segment in segments)
    buffer = AlignedHostBuffer(record_bytes, 8)
    for segment in segments:
        row = getattr(scalar_scales, segment.name)[1]
        payload = bytes(row.contiguous().reshape(-1).view(torch.uint8).tolist())
        ctypes.memmove(buffer.address + segment.offset, payload, len(payload))

    released = []
    completed = DirectRead(ExpertIdentity(0, 1), buffer, released.append)
    destination = MarlinExpertTensors(
        **{
            name: torch.zeros_like(getattr(scalar_scales, name))
            for name in (
                "w13_qweight",
                "w2_qweight",
                "w13_scales",
                "w2_scales",
                "w13_global_scale",
                "w2_global_scale",
            )
        }
    )
    publication = MarlinRecordPublisher(segments).publish(
        completed, destination, slot_id=0
    )

    assert publication.ready()
    assert released == [buffer]
    assert destination.w13_global_scale[0] == scalar_scales.w13_global_scale[1]
    assert destination.w2_global_scale[0] == scalar_scales.w2_global_scale[1]
    buffer.close()
