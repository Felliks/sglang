from __future__ import annotations

from collections.abc import Mapping

from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity


class MemoryExpertStorage:
    """Immutable in-memory records used to validate cache/scheduler lifecycle."""

    def __init__(self, records: Mapping[ExpertIdentity, bytes]) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        sizes = {len(payload) for payload in records.values()}
        if len(sizes) != 1 or next(iter(sizes)) <= 0:
            raise ValueError("all expert records must have one positive size")
        self._records = dict(records)
        self._record_bytes = next(iter(sizes))

    @property
    def record_bytes(self) -> int:
        return self._record_bytes

    def read_into(self, identity: ExpertIdentity, destination: memoryview) -> None:
        if destination.readonly:
            raise ValueError("destination must be writable")
        if destination.nbytes != self._record_bytes:
            raise ValueError(
                f"destination has {destination.nbytes} bytes, expected {self._record_bytes}"
            )
        try:
            payload = self._records[identity]
        except KeyError as exc:
            raise KeyError(f"unknown expert record: {identity}") from exc
        destination[:] = payload
