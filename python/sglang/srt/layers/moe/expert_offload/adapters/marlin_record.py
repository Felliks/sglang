from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertTensors,
)
from sglang.srt.layers.moe.expert_offload.storage.manifest import TensorSegment
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import DirectRead

_TENSOR_NAMES = (
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_global_scale",
    "w2_global_scale",
)


def _row(tensors: MarlinExpertTensors, name: str, index: int) -> torch.Tensor | None:
    tensor = getattr(tensors, name)
    return None if tensor is None else tensor[index]


def marlin_tensor_segments(tensors: MarlinExpertTensors) -> tuple[TensorSegment, ...]:
    """Describe the byte-exact per-expert layout used by a cold store."""

    num_experts = tensors.w13_qweight.shape[0]
    segments = []
    offset = 0
    for name in _TENSOR_NAMES:
        row = _row(tensors, name, 0)
        if row is None:
            continue
        source = getattr(tensors, name)
        if source.shape[0] != num_experts or not row.is_contiguous():
            raise ValueError(f"{name} must contain contiguous per-expert rows")
        nbytes = row.numel() * row.element_size()
        segments.append(
            TensorSegment(
                name=name,
                offset=offset,
                nbytes=nbytes,
                dtype=str(row.dtype).removeprefix("torch."),
                shape=tuple(row.shape),
            )
        )
        offset += nbytes
    return tuple(segments)


def marlin_packing_fingerprint(segments: tuple[TensorSegment, ...]) -> str:
    payload = json.dumps(
        [asdict(segment) for segment in segments],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class PendingMarlinPublication:
    """Keeps a direct-read buffer leased until CUDA has consumed its bytes."""

    def __init__(self, completed: DirectRead, event: torch.cuda.Event | None) -> None:
        self._completed = completed
        self._event = event
        self._released = False

    def ready(self) -> bool:
        return self._event is None or self._event.query()

    def release_if_ready(self) -> bool:
        if self._released:
            return True
        if not self.ready():
            return False
        self._completed.release()
        self._released = True
        return True

    def wait(self) -> None:
        if self._event is not None:
            self._event.synchronize()
        self.release_if_ready()


class MarlinRecordPublisher:
    """Publish one manifest-described record into one physical Marlin slot."""

    def __init__(self, segments: tuple[TensorSegment, ...]) -> None:
        if not segments:
            raise ValueError("Marlin record segments cannot be empty")
        unknown = {segment.name for segment in segments} - set(_TENSOR_NAMES)
        if unknown:
            raise ValueError(f"unknown Marlin tensor segments: {sorted(unknown)}")
        self._segments = segments

    def publish(
        self,
        completed: DirectRead,
        destination: MarlinExpertTensors,
        slot_id: int,
    ) -> PendingMarlinPublication:
        device = destination.w13_qweight.device
        for segment in self._segments:
            row = _row(destination, segment.name, slot_id)
            if row is None:
                raise ValueError(f"destination lacks tensor segment {segment.name}")
            if tuple(row.shape) != segment.shape:
                raise ValueError(f"shape mismatch for tensor segment {segment.name}")
            if str(row.dtype).removeprefix("torch.") != segment.dtype:
                raise ValueError(f"dtype mismatch for tensor segment {segment.name}")
            if not row.is_contiguous():
                raise ValueError(f"destination row is not contiguous: {segment.name}")
            destination_bytes = row.view(torch.uint8).reshape(-1)
            if destination_bytes.numel() != segment.nbytes:
                raise ValueError(
                    f"byte-size mismatch for tensor segment {segment.name}"
                )
            source_bytes = torch.frombuffer(
                completed.view(),
                dtype=torch.uint8,
                count=segment.nbytes,
                offset=segment.offset,
            )
            destination_bytes.copy_(source_bytes, non_blocking=device.type == "cuda")

        event = None
        if device.type == "cuda":
            event = torch.cuda.Event(blocking=False)
            event.record(torch.cuda.current_stream(device))
        publication = PendingMarlinPublication(completed, event)
        if event is None:
            publication.release_if_ready()
        return publication


class BoundMarlinRecordPublisher:
    """Bind the generic scheduler publisher contract to one layer's slots."""

    def __init__(
        self,
        destination: MarlinExpertTensors,
        segments: tuple[TensorSegment, ...],
    ) -> None:
        self.destination = destination
        self._publisher = MarlinRecordPublisher(segments)

    def publish(
        self,
        completed: DirectRead,
        slot_id: int,
    ) -> PendingMarlinPublication:
        return self._publisher.publish(completed, self.destination, slot_id)
