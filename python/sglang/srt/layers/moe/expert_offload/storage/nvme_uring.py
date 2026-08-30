from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
import threading
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Self

from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.storage.manifest import (
    ExpertStoreManifest,
)
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    AlignedHostBuffer,
    DirectRead,
    HostBufferRegistration,
)


class _NativeIoUring:
    """Small C-ABI binding; all ring access stays on one Python thread."""

    def __init__(
        self,
        library_path: Path | None,
        data_path: Path,
        buffers: list[AlignedHostBuffer],
    ) -> None:
        selected = (
            str(library_path)
            if library_path is not None
            else os.environ.get("SGLANG_EXPERT_IO_URING_LIBRARY")
            or ctypes.util.find_library("sglang_expert_io_uring")
        )
        if not selected:
            raise RuntimeError(
                "io_uring expert storage needs libsglang_expert_io_uring; "
                "set --expert-io-uring-library or "
                "SGLANG_EXPERT_IO_URING_LIBRARY"
            )
        self._library = ctypes.CDLL(selected, use_errno=True)
        self._library.sglang_expert_uring_abi_version.argtypes = []
        self._library.sglang_expert_uring_abi_version.restype = ctypes.c_uint32
        abi_version = self._library.sglang_expert_uring_abi_version()
        if abi_version != 1:
            raise RuntimeError(f"unsupported expert io_uring helper ABI: {abi_version}")
        self._library.sglang_expert_uring_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.sglang_expert_uring_create.restype = ctypes.c_int
        self._library.sglang_expert_uring_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        self._library.sglang_expert_uring_submit.restype = ctypes.c_int
        self._library.sglang_expert_uring_wait.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.sglang_expert_uring_wait.restype = ctypes.c_int
        self._library.sglang_expert_uring_destroy.argtypes = [ctypes.c_void_p]
        self._library.sglang_expert_uring_destroy.restype = None
        addresses = (ctypes.c_void_p * len(buffers))(
            *(buffer.address for buffer in buffers)
        )
        self._handle = ctypes.c_void_p()
        result = self._library.sglang_expert_uring_create(
            os.fsencode(data_path),
            addresses,
            buffers[0].size,
            len(buffers),
            ctypes.byref(self._handle),
        )
        if result < 0:
            raise OSError(-result, os.strerror(-result), data_path)
        self._closed = False

    def submit(self, index: int, buffer: AlignedHostBuffer, offset: int) -> None:
        result = self._library.sglang_expert_uring_submit(
            self._handle,
            index,
            buffer.size,
            offset,
        )
        if result < 0:
            raise OSError(-result, os.strerror(-result))

    def wait(self) -> tuple[int, int]:
        index = ctypes.c_uint()
        read_result = ctypes.c_int()
        result = self._library.sglang_expert_uring_wait(
            self._handle, ctypes.byref(index), ctypes.byref(read_result)
        )
        if result < 0:
            raise OSError(-result, os.strerror(-result))
        return index.value, read_result.value

    def close(self) -> None:
        if not self._closed:
            self._library.sglang_expert_uring_destroy(self._handle)
            self._closed = True


class IoUringExpertStorage:
    """Bounded O_DIRECT reader backed by one fixed-buffer io_uring.

    The ring has a single issuer.  Callers enqueue requests and receive normal
    ``Future[ExpertRecordLease]`` objects, keeping the scheduler independent of
    the storage implementation.  Completed buffers remain leased until the
    asynchronous device publication releases them.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        io_depth: int = 2,
        verify_checksums: bool = False,
        buffer_registrar=None,
        library_path: Path | None = None,
        _native_factory=_NativeIoUring,
    ) -> None:
        if io_depth <= 0:
            raise ValueError("io_depth must be positive")
        self.manifest = ExpertStoreManifest.load(manifest_path)
        self._verify_checksums = verify_checksums
        if verify_checksums and self.manifest.record_sha256 is None:
            raise ValueError("checksum verification requested but manifest has none")
        data_path = manifest_path.parent / self.manifest.data_file
        actual_bytes = data_path.stat().st_size
        if actual_bytes != self.manifest.file_bytes:
            raise ValueError(
                f"expert data file has {actual_bytes} bytes, expected "
                f"{self.manifest.file_bytes}"
            )

        self._buffers: list[AlignedHostBuffer] = []
        self._registrations: list[HostBufferRegistration] = []
        self._native = None
        try:
            for _ in range(io_depth):
                buffer = AlignedHostBuffer(
                    self.manifest.record_bytes, self.manifest.alignment
                )
                self._buffers.append(buffer)
                if buffer_registrar is not None:
                    self._registrations.append(buffer_registrar(buffer))
        except BaseException:
            for registration in reversed(self._registrations):
                registration.close()
            for buffer in self._buffers:
                buffer.close()
            raise

        self._condition = threading.Condition()
        self._pending: deque[tuple[ExpertIdentity, Future[DirectRead]]] = deque()
        self._available = deque(range(io_depth))
        self._inflight: dict[int, tuple[ExpertIdentity, Future[DirectRead]]] = {}
        self._leased = 0
        self._shutdown = False
        self._closed = False
        self._native_factory = _native_factory
        self._library_path = library_path
        self._data_path = data_path
        self._startup_done = False
        self._startup_error: BaseException | None = None
        self._worker = threading.Thread(
            target=self._run, name="expert-io-uring", daemon=True
        )
        self._worker.start()
        with self._condition:
            while not self._startup_done:
                self._condition.wait()
            startup_error = self._startup_error
        if startup_error is not None:
            self._worker.join()
            for registration in reversed(self._registrations):
                registration.close()
            for buffer in self._buffers:
                buffer.close()
            raise startup_error

    @property
    def record_bytes(self) -> int:
        return self.manifest.record_bytes

    def _return_buffer(self, index: int) -> None:
        with self._condition:
            self._leased -= 1
            self._available.append(index)
            self._condition.notify()

    def _complete(self, index: int, read_result: int) -> None:
        with self._condition:
            identity, future = self._inflight.pop(index)
        buffer = self._buffers[index]
        if read_result != buffer.size:
            error = (
                OSError(-read_result, os.strerror(-read_result))
                if read_result < 0
                else OSError(f"short expert read: {read_result} of {buffer.size} bytes")
            )
            with self._condition:
                self._available.append(index)
                self._condition.notify()
            future.set_exception(error)
            return
        expected = self.manifest.checksum(identity)
        if self._verify_checksums and expected is not None:
            actual = hashlib.sha256(buffer.view()).hexdigest()
            if actual != expected:
                with self._condition:
                    self._available.append(index)
                    self._condition.notify()
                future.set_exception(
                    OSError(f"expert record checksum mismatch: {identity}")
                )
                return
        with self._condition:
            self._leased += 1
        future.set_result(
            DirectRead(identity, buffer, lambda unused: self._return_buffer(index))
        )

    @staticmethod
    def _fail_future(future: Future[DirectRead], error: BaseException) -> None:
        if not future.cancelled() and not future.done():
            future.set_exception(error)

    def _run(self) -> None:
        try:
            native = self._native_factory(
                self._library_path, self._data_path, self._buffers
            )
        except BaseException as error:
            with self._condition:
                self._startup_error = error
                self._startup_done = True
                self._shutdown = True
                self._condition.notify_all()
            return
        with self._condition:
            self._native = native
            self._startup_done = True
            self._condition.notify_all()
        while True:
            submission = None
            wait_for_completion = False
            with self._condition:
                if self._shutdown:
                    error = RuntimeError("expert storage closed before submission")
                    while self._pending:
                        _, future = self._pending.popleft()
                        self._fail_future(future, error)
                if not self._shutdown and self._pending and self._available:
                    identity, future = self._pending.popleft()
                    index = self._available.popleft()
                    if future.set_running_or_notify_cancel():
                        self._inflight[index] = (identity, future)
                        submission = (index, identity, future)
                    else:
                        self._available.append(index)
                elif self._inflight:
                    wait_for_completion = True
                elif self._shutdown:
                    return
                else:
                    self._condition.wait()
                    continue

            if submission is not None:
                index, identity, future = submission
                try:
                    self._native.submit(
                        index,
                        self._buffers[index],
                        self.manifest.record_offset(identity),
                    )
                except BaseException as error:
                    with self._condition:
                        self._inflight.pop(index)
                        self._available.append(index)
                        self._condition.notify()
                    future.set_exception(error)
                continue
            if wait_for_completion:
                try:
                    index, read_result = self._native.wait()
                    self._complete(index, read_result)
                except BaseException as error:
                    with self._condition:
                        failed = list(self._inflight.items())
                        self._inflight.clear()
                        self._available.extend(index for index, _ in failed)
                        self._shutdown = True
                        self._condition.notify_all()
                    for _, (_, future) in failed:
                        self._fail_future(future, error)

    def submit(self, identity: ExpertIdentity) -> Future[DirectRead]:
        self.manifest.record_index(identity)
        future: Future[DirectRead] = Future()
        with self._condition:
            if self._closed or self._shutdown:
                raise RuntimeError("expert storage is closed")
            self._pending.append((identity, future))
            self._condition.notify()
        return future

    def read_into(self, identity: ExpertIdentity, destination: memoryview) -> None:
        if destination.readonly or destination.nbytes != self.record_bytes:
            raise ValueError("destination must be one writable expert record")
        with self.submit(identity).result() as completed:
            destination[:] = completed.view()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._shutdown = True
            self._condition.notify_all()
        self._worker.join()
        with self._condition:
            if self._leased or len(self._available) != len(self._buffers):
                raise RuntimeError(
                    "cannot close expert storage with leased read buffers"
                )
        self._native.close()
        for registration in reversed(self._registrations):
            registration.close()
        for buffer in self._buffers:
            buffer.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
