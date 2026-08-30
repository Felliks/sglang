from __future__ import annotations

from typing import Self

import torch
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    AlignedHostBuffer,
)


def _cuda_error_value(result) -> int:
    return int(getattr(result, "value", result))


class CudaHostRegistration:
    """Page-lock one aligned I/O buffer for asynchronous CUDA publication."""

    def __init__(self, buffer: AlignedHostBuffer) -> None:
        self._buffer = buffer
        self._runtime = torch.cuda.cudart()
        result = self._runtime.cudaHostRegister(buffer.address, buffer.size, 0)
        if _cuda_error_value(result) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {result}")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        result = self._runtime.cudaHostUnregister(self._buffer.address)
        if _cuda_error_value(result) != 0:
            raise RuntimeError(f"cudaHostUnregister failed: {result}")
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
