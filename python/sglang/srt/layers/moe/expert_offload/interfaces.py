from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass(frozen=True, order=True)
class ExpertIdentity:
    layer_id: int
    expert_id: int


class AdmissionKind(str, Enum):
    OWNER = "owner"
    INFLIGHT = "inflight"
    RESIDENT = "resident"


@dataclass(frozen=True)
class Admission:
    """A generation-qualified claim on a slot.

    Only the OWNER that transitioned the slot to LOADING may publish or fail it.
    Other callers receive INFLIGHT and wait through the scheduler; RESIDENT needs
    no storage operation.
    """

    identity: ExpertIdentity
    slot_id: int
    generation: int
    kind: AdmissionKind
    evicted: ExpertIdentity | None = None


@dataclass(frozen=True)
class SlotLease:
    identity: ExpertIdentity
    slot_id: int
    generation: int


class ExpertStorageBackend(Protocol):
    @property
    def record_bytes(self) -> int: ...

    def read_into(self, identity: ExpertIdentity, destination: memoryview) -> None: ...


class ExpertCacheBackend(Protocol):
    def admit(
        self,
        identity: ExpertIdentity,
        *,
        protected: frozenset[ExpertIdentity] = frozenset(),
    ) -> Admission: ...

    def publish(self, admission: Admission) -> None: ...

    def fail(self, admission: Admission) -> None: ...

    def pin(self, identity: ExpertIdentity) -> SlotLease | None: ...

    def unpin(self, lease: SlotLease) -> None: ...
