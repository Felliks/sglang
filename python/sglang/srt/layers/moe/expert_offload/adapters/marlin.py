from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from sglang.srt.layers.moe.expert_offload.cache import BoundedExpertCache
from sglang.srt.layers.moe.expert_offload.interfaces import (
    AdmissionKind,
    ExpertIdentity,
    SlotLease,
)


@dataclass(frozen=True)
class MarlinExpertTensors:
    """Marlin tensors whose leading dimension addresses physical experts."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_global_scale: torch.Tensor | None = None
    w2_global_scale: torch.Tensor | None = None


class PreparedMarlinExperts:
    """Pinned resident view for exactly one Marlin kernel submission."""

    def __init__(
        self,
        adapter: MarlinExpertOffloadAdapter,
        leases: list[SlotLease],
    ) -> None:
        self.tensors = adapter.resident_tensors
        self.expert_map = adapter.expert_map
        self.global_num_experts = adapter.num_experts
        self._adapter = adapter
        self._leases = leases
        self._completed = False

    def record_completion(self) -> None:
        if self._completed:
            raise RuntimeError("Marlin expert submission was already completed")
        self._adapter._record_completion(self._leases)
        self._completed = True


class MarlinExpertOffloadAdapter:
    """Bounded physical slots backed by a full Marlin tensor bundle.

    This first-stage adapter validates logical residency and numerical parity.
    The backing tensors remain on their original device, so it does not reclaim
    memory yet.  A storage-backed loader can replace ``_copy_expert`` without
    changing the cache lifecycle or Marlin runner contract.

    GPU submissions are deliberately restricted to one CUDA stream.  Updating
    a shared expert map from multiple streams would race an earlier asynchronous
    kernel; unsupported concurrency fails closed instead of risking corruption.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        capacity: int,
        backing_tensors: MarlinExpertTensors,
    ) -> None:
        self.layer_id = layer_id
        self.num_experts = backing_tensors.w13_qweight.shape[0]
        if not 0 < capacity <= self.num_experts:
            raise ValueError("capacity must be in [1, num_experts]")
        self._validate_bundle(backing_tensors)
        self._backing_tensors = backing_tensors
        self.resident_tensors = MarlinExpertTensors(
            w13_qweight=self._allocate_slots(backing_tensors.w13_qweight, capacity),
            w2_qweight=self._allocate_slots(backing_tensors.w2_qweight, capacity),
            w13_scales=self._allocate_slots(backing_tensors.w13_scales, capacity),
            w2_scales=self._allocate_slots(backing_tensors.w2_scales, capacity),
            w13_global_scale=self._allocate_optional_slots(
                backing_tensors.w13_global_scale, capacity
            ),
            w2_global_scale=self._allocate_optional_slots(
                backing_tensors.w2_global_scale, capacity
            ),
        )
        device = backing_tensors.w13_qweight.device
        self.expert_map = torch.full(
            (self.num_experts,), -1, dtype=torch.int32, device=device
        )
        self._cache = BoundedExpertCache(capacity)
        self._pending: list[tuple[torch.cuda.Event, list[SlotLease]]] = []
        self._cuda_stream: int | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _allocate_slots(source: torch.Tensor, capacity: int) -> torch.Tensor:
        return torch.empty(
            (capacity, *source.shape[1:]),
            dtype=source.dtype,
            device=source.device,
        )

    def _allocate_optional_slots(
        self, source: torch.Tensor | None, capacity: int
    ) -> torch.Tensor | None:
        if source is None:
            return None
        if source.ndim == 0 or source.shape[0] != self.num_experts:
            raise ValueError("Marlin global scales must have one row per expert")
        return self._allocate_slots(source, capacity)

    def _validate_bundle(self, tensors: MarlinExpertTensors) -> None:
        required = (
            tensors.w13_qweight,
            tensors.w2_qweight,
            tensors.w13_scales,
            tensors.w2_scales,
        )
        device = required[0].device
        for tensor in required:
            if tensor.ndim == 0 or tensor.shape[0] != self.num_experts:
                raise ValueError(
                    "every Marlin expert tensor needs one leading row per expert"
                )
            if tensor.device != device:
                raise ValueError("all Marlin expert tensors must share one device")
        for tensor in (tensors.w13_global_scale, tensors.w2_global_scale):
            if tensor is not None and tensor.device != device:
                raise ValueError("all Marlin expert tensors must share one device")

    def _identity(self, expert_id: int) -> ExpertIdentity:
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f"expert id {expert_id} is outside the layer")
        return ExpertIdentity(self.layer_id, expert_id)

    def _check_stream(self) -> None:
        device = self._backing_tensors.w13_qweight.device
        if device.type != "cuda":
            return
        stream_id = torch.cuda.current_stream(device).cuda_stream
        if self._cuda_stream is None:
            self._cuda_stream = stream_id
        elif stream_id != self._cuda_stream:
            raise RuntimeError(
                "Marlin expert offload currently requires one CUDA stream"
            )

    def _reap_completed(self) -> None:
        remaining: list[tuple[torch.cuda.Event, list[SlotLease]]] = []
        for event, leases in self._pending:
            if event.query():
                for lease in leases:
                    self._cache.unpin(lease)
            else:
                remaining.append((event, leases))
        self._pending = remaining

    @staticmethod
    def _copy_row(
        destination: torch.Tensor | None,
        source: torch.Tensor | None,
        slot_id: int,
        expert_id: int,
    ) -> None:
        if destination is not None:
            assert source is not None
            destination[slot_id].copy_(source[expert_id], non_blocking=True)

    def _copy_expert(self, expert_id: int, slot_id: int) -> None:
        source = self._backing_tensors
        destination = self.resident_tensors
        self._copy_row(destination.w13_qweight, source.w13_qweight, slot_id, expert_id)
        self._copy_row(destination.w2_qweight, source.w2_qweight, slot_id, expert_id)
        self._copy_row(destination.w13_scales, source.w13_scales, slot_id, expert_id)
        self._copy_row(destination.w2_scales, source.w2_scales, slot_id, expert_id)
        self._copy_row(
            destination.w13_global_scale,
            source.w13_global_scale,
            slot_id,
            expert_id,
        )
        self._copy_row(
            destination.w2_global_scale,
            source.w2_global_scale,
            slot_id,
            expert_id,
        )

    def prepare(self, topk_ids: torch.Tensor) -> PreparedMarlinExperts:
        with self._lock:
            self._check_stream()
            self._reap_completed()
            expert_ids = sorted(set(topk_ids.detach().reshape(-1).cpu().tolist()))
            identities = [self._identity(expert_id) for expert_id in expert_ids]
            if len(identities) > self._cache.capacity:
                raise RuntimeError(
                    "one Marlin submission demands more experts than the resident cache"
                )
            protected = frozenset(identities)

            for identity in identities:
                admission = self._cache.admit(identity, protected=protected)
                if admission.kind == AdmissionKind.INFLIGHT:
                    raise RuntimeError("an expert load is still in flight")
                if admission.kind == AdmissionKind.RESIDENT:
                    continue
                try:
                    if admission.evicted is not None:
                        self.expert_map[admission.evicted.expert_id] = -1
                    self._copy_expert(identity.expert_id, admission.slot_id)
                    self.expert_map[identity.expert_id] = admission.slot_id
                    self._cache.publish(admission)
                except BaseException:
                    self.expert_map[identity.expert_id] = -1
                    self._cache.fail(admission)
                    raise

            leases: list[SlotLease] = []
            try:
                for identity in identities:
                    lease = self._cache.pin(identity)
                    if lease is None:
                        raise RuntimeError("resident expert vanished before pin")
                    leases.append(lease)
            except BaseException:
                for lease in leases:
                    self._cache.unpin(lease)
                raise
            return PreparedMarlinExperts(self, leases)

    def _record_completion(self, leases: list[SlotLease]) -> None:
        with self._lock:
            device = self._backing_tensors.w13_qweight.device
            if device.type == "cuda":
                event = torch.cuda.Event(blocking=False)
                event.record(torch.cuda.current_stream(device))
                self._pending.append((event, leases))
            else:
                for lease in leases:
                    self._cache.unpin(lease)
