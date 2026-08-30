from __future__ import annotations

import ctypes
import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Self

import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertTensors,
)
from sglang.srt.layers.moe.expert_offload.adapters.marlin_record import (
    marlin_packing_fingerprint,
    marlin_tensor_segments,
)
from sglang.srt.layers.moe.expert_offload.storage.cuda_host import (
    CudaHostRegistration,
)
from sglang.srt.layers.moe.expert_offload.storage.manifest import (
    ExpertStoreManifest,
)
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    _LIBC,
    AlignedHostBuffer,
    HostBufferRegistration,
)

_LIBC.pwrite.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_longlong,
]
_LIBC.pwrite.restype = ctypes.c_ssize_t
_LIBC.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_LIBC.memset.restype = ctypes.c_void_p


class ExpertStoreWriter:
    """Crash-safe sequential writer for one immutable expert cold store."""

    def __init__(
        self,
        manifest_path: Path,
        manifest: ExpertStoreManifest,
        *,
        direct: bool = True,
        buffer_registrar: (
            Callable[[AlignedHostBuffer], HostBufferRegistration] | None
        ) = None,
    ) -> None:
        if manifest.record_sha256 is not None:
            raise ValueError("a new store manifest cannot contain record checksums")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.data_path = manifest_path.parent / manifest.data_file
        self.partial_data_path = self.data_path.with_name(
            self.data_path.name + ".partial"
        )
        self.partial_manifest_path = manifest_path.with_name(
            manifest_path.name + ".partial"
        )
        for path in (
            self.manifest_path,
            self.data_path,
            self.partial_manifest_path,
            self.partial_data_path,
        ):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite expert store path: {path}"
                )

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if direct:
            direct_flag = getattr(os, "O_DIRECT", None)
            if direct_flag is None:
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= direct_flag
        self._fd = os.open(self.partial_data_path, flags, 0o644)
        self._buffer = AlignedHostBuffer(manifest.record_bytes, manifest.alignment)
        self._registration = (
            buffer_registrar(self._buffer) if buffer_registrar is not None else None
        )
        self._checksums: list[str] = []
        self._next_record = 0
        self._closed = False
        self._finished = False

    @property
    def buffer(self) -> AlignedHostBuffer:
        return self._buffer

    def write_record(self, fill: Callable[[memoryview], None]) -> None:
        if self._closed or self._finished:
            raise RuntimeError("expert store writer is closed")
        if self._next_record >= self.manifest.num_records:
            raise RuntimeError("expert store already contains every record")
        _LIBC.memset(
            ctypes.c_void_p(self._buffer.address),
            0,
            self._buffer.size,
        )
        fill(self._buffer.view())
        checksum = hashlib.sha256(self._buffer.view()).hexdigest()
        result = _LIBC.pwrite(
            self._fd,
            ctypes.c_void_p(self._buffer.address),
            self._buffer.size,
            self._next_record * self._buffer.size,
        )
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if result != self._buffer.size:
            raise OSError(f"short expert write: {result} of {self._buffer.size} bytes")
        self._checksums.append(checksum)
        self._next_record += 1

    def finish(self) -> ExpertStoreManifest:
        if self._closed or self._finished:
            raise RuntimeError("expert store writer is closed")
        if self._next_record != self.manifest.num_records:
            raise RuntimeError(
                f"expert store has {self._next_record} of "
                f"{self.manifest.num_records} records"
            )
        os.fsync(self._fd)
        self._close_resources()
        os.rename(self.partial_data_path, self.data_path)
        completed_manifest = replace(
            self.manifest, record_sha256=tuple(self._checksums)
        )
        completed_manifest.save(self.partial_manifest_path)
        manifest_fd = os.open(self.partial_manifest_path, os.O_RDONLY)
        try:
            os.fsync(manifest_fd)
        finally:
            os.close(manifest_fd)
        os.rename(self.partial_manifest_path, self.manifest_path)
        directory_fd = os.open(self.manifest_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._finished = True
        return completed_manifest

    def _close_resources(self) -> None:
        if self._closed:
            return
        if self._registration is not None:
            self._registration.close()
        self._buffer.close()
        os.close(self._fd)
        self._closed = True

    def abort(self) -> None:
        """Close resources but keep partial files as explicit crash evidence."""

        self._close_resources()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._finished:
            self.abort()


def _copy_marlin_expert_to_record(
    tensors: MarlinExpertTensors,
    expert_id: int,
    segments,
    destination: memoryview,
) -> None:
    device = tensors.w13_qweight.device
    for segment in segments:
        row = getattr(tensors, segment.name)[expert_id]
        source_bytes = row.contiguous().reshape(-1).view(torch.uint8)
        if source_bytes.numel() != segment.nbytes:
            raise ValueError(f"Marlin record byte-size changed: {segment.name}")
        destination_bytes = torch.frombuffer(
            destination,
            dtype=torch.uint8,
            count=segment.nbytes,
            offset=segment.offset,
        )
        destination_bytes.copy_(source_bytes, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def build_marlin_expert_store(
    manifest_path: Path,
    layers: Sequence[MarlinExpertTensors],
    *,
    model_fingerprint: str,
    data_file: str = "experts.marlin.bin",
    alignment: int = 4096,
    direct: bool = True,
) -> ExpertStoreManifest:
    """Serialize already-packed Marlin expert rows without changing quantization."""

    if not layers:
        raise ValueError("Marlin expert layers cannot be empty")
    segments = marlin_tensor_segments(layers[0])
    experts_per_layer = layers[0].w13_qweight.shape[0]
    for layer in layers:
        if layer.w13_qweight.shape[0] != experts_per_layer:
            raise ValueError("all Marlin layers must contain the same expert count")
        if marlin_tensor_segments(layer) != segments:
            raise ValueError("all Marlin layers must share one record layout")
    raw_bytes = max(segment.offset + segment.nbytes for segment in segments)
    record_bytes = (raw_bytes + alignment - 1) // alignment * alignment
    manifest = ExpertStoreManifest(
        data_file=data_file,
        alignment=alignment,
        record_bytes=record_bytes,
        num_layers=len(layers),
        experts_per_layer=experts_per_layer,
        tensor_segments=segments,
        model_fingerprint=model_fingerprint,
        packing_fingerprint=marlin_packing_fingerprint(segments),
    )
    device = layers[0].w13_qweight.device
    if any(layer.w13_qweight.device != device for layer in layers):
        raise ValueError("all Marlin layers must share one device")
    registrar = CudaHostRegistration if device.type == "cuda" else None
    with ExpertStoreWriter(
        manifest_path,
        manifest,
        direct=direct,
        buffer_registrar=registrar,
    ) as writer:
        for tensors in layers:
            for expert_id in range(experts_per_layer):
                writer.write_record(
                    lambda destination, tensors=tensors, expert_id=expert_id: (
                        _copy_marlin_expert_to_record(
                            tensors,
                            expert_id,
                            segments,
                            destination,
                        )
                    )
                )
        return writer.finish()
