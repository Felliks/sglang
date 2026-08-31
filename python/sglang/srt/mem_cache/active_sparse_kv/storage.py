from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Self, Sequence

import torch
from sglang.srt.mem_cache.active_sparse_kv.config import ActiveSparseKVConfig
from sglang.srt.mem_cache.active_sparse_kv.layout import ActiveSparseKVLayout


class ActiveKVExtent:
    """Process-owned, preallocated ephemeral extent with bounded disk use."""

    def __init__(
        self, config: ActiveSparseKVConfig, layout: ActiveSparseKVLayout, rank: int
    ) -> None:
        if config.backend != "nvme" or config.path is None:
            raise ValueError("ActiveKVExtent requires the NVMe backend")
        layout.validate_capacity(config.max_bytes)
        directory = config.path
        directory.mkdir(parents=True, exist_ok=True)
        # A SIGKILL cannot execute ``close``.  Lock every live extent and
        # reclaim only files whose advisory lock can be acquired, which is
        # namespace-independent and cannot delete another running rank's data.
        for candidate in directory.glob("active-kv-rank*-pid*.bin"):
            stale_fd = os.open(candidate, os.O_RDWR | os.O_CLOEXEC)
            try:
                try:
                    fcntl.flock(stale_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                candidate.unlink(missing_ok=True)
            finally:
                os.close(stale_fd)
        free_bytes = shutil.disk_usage(directory).free
        if free_bytes - layout.file_bytes < config.min_free_bytes:
            raise OSError(
                "active sparse-KV extent would cross the configured free-space "
                f"floor: free={free_bytes}, required={layout.file_bytes}, "
                f"floor={config.min_free_bytes}"
            )
        self.path = directory / f"active-kv-rank{rank}-pid{os.getpid()}.bin"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        fd = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.posix_fallocate(fd, 0, layout.file_bytes)
        except BaseException:
            os.close(fd)
            self.path.unlink(missing_ok=True)
            raise
        self._lock_fd = fd
        self.file_bytes = layout.file_bytes
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.path.unlink(missing_ok=True)
            os.close(self._lock_fd)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class NativeActiveKVUring:
    """Single-issuer O_DIRECT ring over fixed CUDA-pinned staging rows."""

    _READ = 0
    _WRITE = 1
    _READ_BUFFERED = 2

    def __init__(
        self,
        path: Path,
        *,
        record_bytes: int,
        io_depth: int,
        page_cache_bytes: int = 0,
        library_path: Path | None = None,
    ) -> None:
        if record_bytes <= 0 or record_bytes % 4096:
            raise ValueError("record_bytes must be a positive 4096-byte multiple")
        if io_depth <= 0:
            raise ValueError("io_depth must be positive")
        selected = (
            str(library_path)
            if library_path is not None
            else os.environ.get("SGLANG_ACTIVE_KV_IO_URING_LIBRARY")
            or ctypes.util.find_library("sglang_active_kv_io_uring")
        )
        if not selected:
            raise RuntimeError(
                "NVMe active sparse-KV needs libsglang_active_kv_io_uring; "
                "set SGLANG_ACTIVE_KV_IO_URING_LIBRARY"
            )
        self._library = ctypes.CDLL(selected, use_errno=True)
        self._bind_abi()
        abi_version = self._library.sglang_active_kv_uring_abi_version()
        if abi_version not in {1, 2}:
            raise RuntimeError(
                f"unsupported active-KV io_uring helper ABI: {abi_version}"
            )
        self.record_bytes = record_bytes
        self.io_depth = io_depth
        self._abi_version = abi_version
        self._page_cache_records = page_cache_bytes // record_bytes
        if self._page_cache_records and abi_version < 2:
            raise RuntimeError(
                "active-KV page-cache path requires io_uring helper ABI 2"
            )
        self._page_cache_lru: OrderedDict[int, None] = OrderedDict()
        self.buffered_reads = 0
        self.direct_reads = 0
        # CUDA's pinned allocator does not promise the filesystem-specific
        # alignment required by O_DIRECT.  Keep the oversized owner alive and
        # expose a 4 KiB-aligned contiguous view to both CUDA and io_uring.
        self._staging_banks = 2
        registered_depth = io_depth * self._staging_banks
        staging_bytes = registered_depth * record_bytes
        self._staging_owner = torch.empty(
            staging_bytes + 4095,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        alignment_offset = (-self._staging_owner.data_ptr()) % 4096
        self.staging = self._staging_owner.narrow(
            0, alignment_offset, staging_bytes
        ).view(registered_depth, record_bytes)
        addresses = []
        for row in self.staging:
            address = row.data_ptr()
            if address % 4096:
                raise RuntimeError(
                    "CUDA-pinned active-KV staging row is not O_DIRECT aligned"
                )
            addresses.append(address)
        native_addresses = (ctypes.c_void_p * registered_depth)(*addresses)
        self._handle = ctypes.c_void_p()
        result = self._library.sglang_active_kv_uring_create(
            os.fsencode(path),
            native_addresses,
            record_bytes,
            registered_depth,
            ctypes.byref(self._handle),
        )
        if result < 0:
            raise OSError(-result, os.strerror(-result), path)
        try:
            self._advice_fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        except BaseException:
            self._library.sglang_active_kv_uring_destroy(self._handle)
            raise
        self._offsets = (ctypes.c_int64 * io_depth)()
        self._completed_indices = (ctypes.c_uint * io_depth)()
        self._results = (ctypes.c_int * io_depth)()
        self._closed = False

    def _bind_abi(self) -> None:
        library = self._library
        library.sglang_active_kv_uring_abi_version.argtypes = []
        library.sglang_active_kv_uring_abi_version.restype = ctypes.c_uint32
        library.sglang_active_kv_uring_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.sglang_active_kv_uring_create.restype = ctypes.c_int
        library.sglang_active_kv_uring_submit_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint,
        ]
        library.sglang_active_kv_uring_submit_batch.restype = ctypes.c_int
        library.sglang_active_kv_uring_wait_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.sglang_active_kv_uring_wait_batch.restype = ctypes.c_int
        library.sglang_active_kv_uring_destroy.argtypes = [ctypes.c_void_p]
        library.sglang_active_kv_uring_destroy.restype = None

    def _transfer(
        self,
        operation: int,
        offsets: Sequence[int],
        buffer_indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        count = len(offsets)
        if not 0 < count <= self.io_depth:
            raise ValueError(
                f"batch must contain 1..{self.io_depth} records, got {count}"
            )
        if buffer_indices is None:
            buffer_indices = range(count)
        if len(buffer_indices) != count:
            raise ValueError("buffer_indices must match offsets")
        native_indices = (ctypes.c_uint * count)()
        for index, (offset, buffer_index) in enumerate(
            zip(offsets, buffer_indices)
        ):
            if offset < 0 or offset % self.record_bytes:
                raise ValueError(f"unaligned active-KV record offset: {offset}")
            if not 0 <= buffer_index < self.io_depth * self._staging_banks:
                raise ValueError(f"active-KV staging index out of range: {buffer_index}")
            self._offsets[index] = offset
            native_indices[index] = buffer_index
        result = self._library.sglang_active_kv_uring_submit_batch(
            self._handle,
            operation,
            native_indices,
            self._offsets,
            count,
        )
        if result < 0:
            raise OSError(-result, os.strerror(-result))
        result = self._library.sglang_active_kv_uring_wait_batch(
            self._handle,
            count,
            self._completed_indices,
            self._results,
        )
        if result < 0:
            raise OSError(-result, os.strerror(-result))
        for item in range(count):
            io_result = self._results[item]
            if io_result != self.record_bytes:
                if io_result < 0:
                    raise OSError(-io_result, os.strerror(-io_result))
                raise OSError(
                    f"short active-KV I/O: {io_result} of {self.record_bytes} bytes"
                )
        return self.staging[:count]

    def read(self, offsets: Sequence[int], *, bank: int = 0) -> torch.Tensor:
        if not 0 <= bank < self._staging_banks:
            raise ValueError(f"invalid active-KV staging bank: {bank}")
        base = bank * self.io_depth
        if not self._page_cache_records:
            self.direct_reads += len(offsets)
            self._transfer(
                self._READ, offsets, range(base, base + len(offsets))
            )
            return self.staging[base : base + len(offsets)]

        buffered_indices = []
        buffered_offsets = []
        direct_indices = []
        direct_offsets = []
        for index, offset in enumerate(offsets):
            staging_index = base + index
            if offset in self._page_cache_lru:
                buffered_indices.append(staging_index)
                buffered_offsets.append(offset)
            else:
                direct_indices.append(staging_index)
                direct_offsets.append(offset)
        if buffered_offsets:
            self._transfer(
                self._READ_BUFFERED, buffered_offsets, buffered_indices
            )
            self.buffered_reads += len(buffered_offsets)
        if direct_offsets:
            self._transfer(self._READ, direct_offsets, direct_indices)
            self.direct_reads += len(direct_offsets)
            for offset in direct_offsets:
                os.posix_fadvise(
                    self._advice_fd,
                    offset,
                    self.record_bytes,
                    os.POSIX_FADV_WILLNEED,
                )
        for offset in offsets:
            self._page_cache_lru.pop(offset, None)
            self._page_cache_lru[offset] = None
        while len(self._page_cache_lru) > self._page_cache_records:
            cold_offset, _ = self._page_cache_lru.popitem(last=False)
            os.posix_fadvise(
                self._advice_fd,
                cold_offset,
                self.record_bytes,
                os.POSIX_FADV_DONTNEED,
            )
        return self.staging[base : base + len(offsets)]

    def write(self, offsets: Sequence[int], records: torch.Tensor) -> None:
        count = len(offsets)
        if records.shape != (count, self.record_bytes):
            raise ValueError(
                "records must be packed uint8 rows shaped "
                f"[{count}, {self.record_bytes}], got {tuple(records.shape)}"
            )
        if records.dtype != torch.uint8:
            raise ValueError("records must use torch.uint8 storage")
        self.staging[:count].copy_(records, non_blocking=False)
        self._transfer(self._WRITE, offsets)
        for offset in offsets:
            self._page_cache_lru.pop(offset, None)
            os.posix_fadvise(
                self._advice_fd,
                offset,
                self.record_bytes,
                os.POSIX_FADV_DONTNEED,
            )

    def close(self) -> None:
        if not self._closed:
            self._library.sglang_active_kv_uring_destroy(self._handle)
            os.close(self._advice_fd)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
