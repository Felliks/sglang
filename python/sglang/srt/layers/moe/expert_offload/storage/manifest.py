from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity

STORE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TensorSegment:
    name: str
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor segment name cannot be empty")
        if self.offset < 0 or self.nbytes <= 0:
            raise ValueError("tensor segment offset/size is invalid")
        if not self.dtype or any(size <= 0 for size in self.shape):
            raise ValueError("tensor segment dtype/shape is invalid")


@dataclass(frozen=True)
class ExpertStoreManifest:
    data_file: str
    alignment: int
    record_bytes: int
    num_layers: int
    experts_per_layer: int
    tensor_segments: tuple[TensorSegment, ...]
    model_fingerprint: str
    packing_fingerprint: str
    record_sha256: tuple[str, ...] | None = None
    version: int = STORE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.version != STORE_FORMAT_VERSION:
            raise ValueError(f"unsupported expert store version: {self.version}")
        if not self.data_file or Path(self.data_file).name != self.data_file:
            raise ValueError("data_file must be one file name beside the manifest")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if self.record_bytes <= 0 or self.record_bytes % self.alignment:
            raise ValueError("record_bytes must be alignment-sized")
        if self.num_layers <= 0 or self.experts_per_layer <= 0:
            raise ValueError("store geometry must be positive")
        if not self.tensor_segments:
            raise ValueError("tensor_segments cannot be empty")
        if not self.model_fingerprint or not self.packing_fingerprint:
            raise ValueError("model and packing fingerprints are required")

        ordered = sorted(self.tensor_segments, key=lambda segment: segment.offset)
        cursor = 0
        names: set[str] = set()
        for segment in ordered:
            if segment.name in names:
                raise ValueError(f"duplicate tensor segment: {segment.name}")
            if segment.offset < cursor:
                raise ValueError("tensor segments overlap")
            if segment.offset + segment.nbytes > self.record_bytes:
                raise ValueError("tensor segment extends beyond its record")
            cursor = segment.offset + segment.nbytes
            names.add(segment.name)

        if self.record_sha256 is not None:
            if len(self.record_sha256) != self.num_records:
                raise ValueError("record checksum count does not match store geometry")
            if any(len(checksum) != 64 for checksum in self.record_sha256):
                raise ValueError("record checksums must be SHA-256 hex digests")

    @property
    def num_records(self) -> int:
        return self.num_layers * self.experts_per_layer

    @property
    def file_bytes(self) -> int:
        return self.num_records * self.record_bytes

    def record_index(self, identity: ExpertIdentity) -> int:
        if not 0 <= identity.layer_id < self.num_layers:
            raise ValueError(f"layer id is outside the store: {identity.layer_id}")
        if not 0 <= identity.expert_id < self.experts_per_layer:
            raise ValueError(f"expert id is outside the store: {identity.expert_id}")
        return identity.layer_id * self.experts_per_layer + identity.expert_id

    def record_offset(self, identity: ExpertIdentity) -> int:
        return self.record_index(identity) * self.record_bytes

    def checksum(self, identity: ExpertIdentity) -> str | None:
        if self.record_sha256 is None:
            return None
        return self.record_sha256[self.record_index(identity)]

    def save(self, path: Path) -> None:
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> ExpertStoreManifest:
        payload = json.loads(path.read_text())
        payload["tensor_segments"] = tuple(
            TensorSegment(**{**item, "shape": tuple(item["shape"])})
            for item in payload["tensor_segments"]
        )
        if payload.get("record_sha256") is not None:
            payload["record_sha256"] = tuple(payload["record_sha256"])
        return cls(**payload)
