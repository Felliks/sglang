from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from sglang.srt.layers.moe.expert_offload.interfaces import (
    Admission,
    AdmissionKind,
    ExpertIdentity,
    SlotLease,
)


class SlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    RESIDENT = "resident"


class ExpertCacheFullError(RuntimeError):
    """Every physical slot is pinned, loading, or explicitly protected."""


@dataclass
class _Slot:
    slot_id: int
    generation: int = 0
    identity: ExpertIdentity | None = None
    state: SlotState = SlotState.EMPTY
    pins: int = 0
    last_access: int = 0


class BoundedExpertCache:
    """Thread-safe slot lifecycle with fail-closed eviction semantics.

    Storage and CUDA publication are deliberately outside this class.  A slot is
    visible to readers only after its generation-qualified OWNER calls publish.
    Pinned or LOADING slots are never eviction candidates; exhaustion raises
    instead of overwriting a weight that a kernel may still consume.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._slots = [_Slot(slot_id=index) for index in range(capacity)]
        self._by_identity: dict[ExpertIdentity, int] = {}
        self._clock = 0
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return len(self._slots)

    def _touch(self, slot: _Slot) -> None:
        self._clock += 1
        slot.last_access = self._clock

    def _existing(self, identity: ExpertIdentity) -> _Slot | None:
        slot_id = self._by_identity.get(identity)
        if slot_id is None:
            return None
        slot = self._slots[slot_id]
        if slot.identity != identity:
            raise RuntimeError("expert-to-slot index is inconsistent")
        return slot

    def admit(
        self,
        identity: ExpertIdentity,
        *,
        protected: frozenset[ExpertIdentity] = frozenset(),
    ) -> Admission:
        with self._lock:
            existing = self._existing(identity)
            if existing is not None:
                self._touch(existing)
                kind = (
                    AdmissionKind.RESIDENT
                    if existing.state == SlotState.RESIDENT
                    else AdmissionKind.INFLIGHT
                )
                return Admission(
                    identity=identity,
                    slot_id=existing.slot_id,
                    generation=existing.generation,
                    kind=kind,
                )

            empty = next(
                (slot for slot in self._slots if slot.state == SlotState.EMPTY), None
            )
            if empty is None:
                candidates = [
                    slot
                    for slot in self._slots
                    if slot.state == SlotState.RESIDENT
                    and slot.pins == 0
                    and slot.identity not in protected
                ]
                if not candidates:
                    raise ExpertCacheFullError("no unpinned expert slot is available")
                empty = min(candidates, key=lambda slot: slot.last_access)

            evicted = empty.identity
            if evicted is not None:
                del self._by_identity[evicted]
            empty.generation += 1
            empty.identity = identity
            empty.state = SlotState.LOADING
            empty.pins = 0
            self._touch(empty)
            self._by_identity[identity] = empty.slot_id
            return Admission(
                identity=identity,
                slot_id=empty.slot_id,
                generation=empty.generation,
                kind=AdmissionKind.OWNER,
                evicted=evicted,
            )

    def _owned_loading_slot(self, admission: Admission) -> _Slot:
        if admission.kind != AdmissionKind.OWNER:
            raise RuntimeError("only an owner admission can change slot state")
        slot = self._slots[admission.slot_id]
        if (
            slot.generation != admission.generation
            or slot.identity != admission.identity
            or slot.state != SlotState.LOADING
        ):
            raise RuntimeError("stale or invalid expert slot admission")
        return slot

    def publish(self, admission: Admission) -> None:
        with self._lock:
            slot = self._owned_loading_slot(admission)
            slot.state = SlotState.RESIDENT
            self._touch(slot)

    def fail(self, admission: Admission) -> None:
        with self._lock:
            slot = self._owned_loading_slot(admission)
            del self._by_identity[admission.identity]
            slot.identity = None
            slot.state = SlotState.EMPTY
            slot.pins = 0
            self._touch(slot)

    def pin(self, identity: ExpertIdentity) -> SlotLease | None:
        with self._lock:
            slot = self._existing(identity)
            if slot is None or slot.state != SlotState.RESIDENT:
                return None
            slot.pins += 1
            self._touch(slot)
            return SlotLease(
                identity=identity,
                slot_id=slot.slot_id,
                generation=slot.generation,
            )

    def unpin(self, lease: SlotLease) -> None:
        with self._lock:
            slot = self._slots[lease.slot_id]
            if slot.generation != lease.generation or slot.identity != lease.identity:
                raise RuntimeError("stale expert slot lease")
            if slot.pins <= 0:
                raise RuntimeError("expert slot lease is not pinned")
            slot.pins -= 1

    def resident_slot(self, identity: ExpertIdentity) -> int | None:
        with self._lock:
            slot = self._existing(identity)
            if slot is None or slot.state != SlotState.RESIDENT:
                return None
            self._touch(slot)
            return slot.slot_id

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "slot_id": slot.slot_id,
                    "generation": slot.generation,
                    "identity": slot.identity,
                    "state": slot.state.value,
                    "pins": slot.pins,
                    "last_access": slot.last_access,
                }
                for slot in self._slots
            ]
