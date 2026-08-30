from __future__ import annotations

import ctypes
import hashlib
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, Self

from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.storage.manifest import (
    ExpertStoreManifest,
)

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.pread.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_longlong,
]
_LIBC.pread.restype = ctypes.c_ssize_t
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None


class AlignedHostBuffer:
    """Page-aligned host allocation suitable for Linux direct I/O."""

    def __init__(self, size: int, alignment: int) -> None:
        pointer = ctypes.c_void_p()
        error = _LIBC.posix_memalign(ctypes.byref(pointer), alignment, size)
        if error:
            raise OSError(error, os.strerror(error))
        self.size = size
        self.alignment = alignment
        self.address = pointer.value
        if self.address is None:
            raise MemoryError("posix_memalign returned a null pointer")
        self._array = (ctypes.c_ubyte * size).from_address(self.address)
        self._closed = False

    def view(self) -> memoryview:
        if self._closed:
            raise RuntimeError("aligned buffer is closed")
        return memoryview(self._array).cast("B")

    def close(self) -> None:
        if not self._closed:
            _LIBC.free(ctypes.c_void_p(self.address))
            self._closed = True

    def __del__(self) -> None:
        self.close()


class HostBufferRegistration(Protocol):
    def close(self) -> None: ...


class DirectRead:
    """A completed read that keeps its bounded staging slot leased."""

    def __init__(
        self,
        identity: ExpertIdentity,
        buffer: AlignedHostBuffer,
        release_buffer,
    ) -> None:
        self.identity = identity
        self.buffer = buffer
        self._release_buffer = release_buffer
        self._released = False

    def view(self) -> memoryview:
        if self._released:
            raise RuntimeError("direct-read buffer was released")
        return self.buffer.view()

    def release(self) -> None:
        if not self._released:
            self._release_buffer(self.buffer)
            self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


class DirectExpertStorage:
    """Bounded-QD fixed-record reader using aligned ``pread`` requests.

    Threaded pread is intentional for the first portable backend: it preserves
    exact queue-depth and direct-I/O semantics without requiring liburing in the
    SGLang image.  The interface can later be implemented by io_uring while the
    manifest, buffer ownership, and scheduler contracts remain unchanged.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        io_depth: int = 2,
        direct: bool = True,
        verify_checksums: bool = False,
        buffer_registrar: (
            Callable[[AlignedHostBuffer], HostBufferRegistration] | None
        ) = None,
    ) -> None:
        if io_depth <= 0:
            raise ValueError("io_depth must be positive")
        self.manifest = ExpertStoreManifest.load(manifest_path)
        self._verify_checksums = verify_checksums
        if verify_checksums and self.manifest.record_sha256 is None:
            raise ValueError("checksum verification requested but manifest has none")

        flags = os.O_RDONLY
        if direct:
            direct_flag = getattr(os, "O_DIRECT", None)
            if direct_flag is None:
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= direct_flag
        self._fd = os.open(manifest_path.parent / self.manifest.data_file, flags)
        actual_bytes = os.fstat(self._fd).st_size
        if actual_bytes != self.manifest.file_bytes:
            os.close(self._fd)
            raise ValueError(
                f"expert data file has {actual_bytes} bytes, expected "
                f"{self.manifest.file_bytes}"
            )

        self._buffers: queue.Queue[AlignedHostBuffer] = queue.Queue(io_depth)
        self._all_buffers: list[AlignedHostBuffer] = []
        self._registrations: list[HostBufferRegistration] = []
        try:
            for _ in range(io_depth):
                buffer = AlignedHostBuffer(
                    self.manifest.record_bytes,
                    self.manifest.alignment,
                )
                self._all_buffers.append(buffer)
                if buffer_registrar is not None:
                    self._registrations.append(buffer_registrar(buffer))
                self._buffers.put(buffer)
        except BaseException:
            for registration in reversed(self._registrations):
                registration.close()
            for buffer in self._all_buffers:
                buffer.close()
            os.close(self._fd)
            raise
        self._executor = ThreadPoolExecutor(
            max_workers=io_depth, thread_name_prefix="expert-direct-read"
        )
        self._closed = False
        self._shutdown = False
        self._close_lock = threading.Lock()

    @property
    def record_bytes(self) -> int:
        return self.manifest.record_bytes

    def _return_buffer(self, buffer: AlignedHostBuffer) -> None:
        self._buffers.put(buffer)

    def _read(self, identity: ExpertIdentity) -> DirectRead:
        while True:
            try:
                buffer = self._buffers.get(timeout=0.1)
                break
            except queue.Empty:
                if self._shutdown:
                    raise RuntimeError(
                        "expert storage closed while waiting for a buffer"
                    )
        try:
            result = _LIBC.pread(
                self._fd,
                ctypes.c_void_p(buffer.address),
                buffer.size,
                self.manifest.record_offset(identity),
            )
            if result < 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            if result != buffer.size:
                raise OSError(f"short expert read: {result} of {buffer.size} bytes")
            expected = self.manifest.checksum(identity)
            if self._verify_checksums and expected is not None:
                actual = hashlib.sha256(buffer.view()).hexdigest()
                if actual != expected:
                    raise OSError(f"expert record checksum mismatch: {identity}")
            return DirectRead(identity, buffer, self._return_buffer)
        except BaseException:
            self._return_buffer(buffer)
            raise

    def submit(self, identity: ExpertIdentity) -> Future[DirectRead]:
        if self._closed or self._shutdown:
            raise RuntimeError("expert storage is closed")
        self.manifest.record_index(identity)
        return self._executor.submit(self._read, identity)

    def read_into(self, identity: ExpertIdentity, destination: memoryview) -> None:
        if destination.readonly or destination.nbytes != self.record_bytes:
            raise ValueError("destination must be one writable expert record")
        with self.submit(identity).result() as completed:
            destination[:] = completed.view()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            if not self._shutdown:
                self._shutdown = True
                self._executor.shutdown(wait=True, cancel_futures=True)
            if self._buffers.qsize() != len(self._all_buffers):
                raise RuntimeError(
                    "cannot close expert storage with leased read buffers"
                )
            for registration in reversed(self._registrations):
                registration.close()
            for buffer in self._all_buffers:
                buffer.close()
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
