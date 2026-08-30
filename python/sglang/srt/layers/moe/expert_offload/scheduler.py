from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sglang.srt.layers.moe.expert_offload.cache import ExpertCacheFullError
from sglang.srt.layers.moe.expert_offload.interfaces import (
    Admission,
    AdmissionKind,
    AsyncExpertStorageBackend,
    ExpertCacheBackend,
    ExpertIdentity,
    ExpertPublication,
    ExpertPublisher,
    ExpertRecordLease,
    SlotLease,
)


@dataclass(frozen=True)
class PrefetchCandidate:
    identity: ExpertIdentity
    score: float
    deadline_ns: int


@dataclass
class _Queued:
    candidate: PrefetchCandidate
    sequence: int


@dataclass
class _Transfer:
    admission: Admission
    future: object
    submitted_ns: int
    predicted_completion_ns: int
    deadline_ns: int | None
    demand: bool
    completed_ns: int | None = None
    record: ExpertRecordLease | None = None
    publication: ExpertPublication | None = None


class DeadlineExpertScheduler:
    """Causal bounded-QD scheduler for speculative and demanded experts.

    The scheduler owns admission and transfer lifecycle but not model tensors.
    Only the model thread may call public methods. Storage workers merely mark
    completion time through Future callbacks; CUDA publication remains on the
    model thread and the native router is always the source of demanded IDs.
    """

    def __init__(
        self,
        *,
        cache: ExpertCacheBackend,
        storage: AsyncExpertStorageBackend,
        publisher: ExpertPublisher,
        io_depth: int,
        initial_service_ns: int,
        on_evict: Callable[[ExpertIdentity], None] = lambda _: None,
        on_resident: Callable[[Admission], None] = lambda _: None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_pending: int = 512,
    ) -> None:
        if io_depth <= 0 or initial_service_ns <= 0 or max_pending <= 0:
            raise ValueError(
                "scheduler bounds and initial service time must be positive"
            )
        self._cache = cache
        self._storage = storage
        self._publisher = publisher
        self._io_depth = io_depth
        self._service_ns = initial_service_ns
        self._on_evict = on_evict
        self._on_resident = on_resident
        self._clock_ns = clock_ns
        self._max_pending = max_pending
        self._owner_thread = threading.get_ident()
        self._sequence = 0
        self._queued: dict[ExpertIdentity, _Queued] = {}
        self._transfers: dict[ExpertIdentity, _Transfer] = {}
        self.metrics = {
            "prefetch_queued": 0,
            "prefetch_rejected_deadline": 0,
            "prefetch_dropped_capacity": 0,
            "prefetch_expired": 0,
            "reads_submitted": 0,
            "reads_deduplicated": 0,
            "demand_accesses": 0,
            "demand_cache_hits": 0,
            "demand_reads": 0,
            "demand_batches": 0,
            "max_demand_batch_experts": 0,
            "demand_batch_misses": 0,
            "max_demand_batch_misses": 0,
            "publication_failures": 0,
        }

    def _check_thread(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("expert scheduler must run on its owning model thread")

    def _feasible(self, candidate: PrefetchCandidate, now_ns: int) -> bool:
        return now_ns + self._service_ns <= candidate.deadline_ns

    def prefetch(self, candidates: Iterable[PrefetchCandidate]) -> None:
        self._check_thread()
        now_ns = self._clock_ns()
        for candidate in candidates:
            if candidate.score <= 0:
                continue
            lease = self._cache.pin(candidate.identity)
            if lease is not None:
                self._cache.unpin(lease)
                continue
            if candidate.identity in self._transfers:
                self.metrics["reads_deduplicated"] += 1
                continue
            if not self._feasible(candidate, now_ns):
                self.metrics["prefetch_rejected_deadline"] += 1
                continue
            existing = self._queued.get(candidate.identity)
            if existing is not None:
                if (candidate.deadline_ns, -candidate.score) < (
                    existing.candidate.deadline_ns,
                    -existing.candidate.score,
                ):
                    existing.candidate = candidate
                self.metrics["reads_deduplicated"] += 1
                continue
            self._sequence += 1
            self._queued[candidate.identity] = _Queued(candidate, self._sequence)
            self.metrics["prefetch_queued"] += 1
        self._trim_pending()
        self._pump(now_ns)

    def _trim_pending(self) -> None:
        if len(self._queued) <= self._max_pending:
            return
        keep = sorted(
            self._queued.values(),
            key=lambda item: (
                item.candidate.deadline_ns,
                -item.candidate.score,
                item.sequence,
            ),
        )[: self._max_pending]
        self.metrics["prefetch_dropped_capacity"] += len(self._queued) - len(keep)
        self._queued = {item.candidate.identity: item for item in keep}

    def _next_candidate(self, now_ns: int) -> PrefetchCandidate | None:
        expired = [
            identity
            for identity, item in self._queued.items()
            if not self._feasible(item.candidate, now_ns)
        ]
        for identity in expired:
            del self._queued[identity]
            self.metrics["prefetch_expired"] += 1
        if not self._queued:
            return None
        item = min(
            self._queued.values(),
            key=lambda queued: (
                queued.candidate.deadline_ns,
                -queued.candidate.score,
                queued.sequence,
            ),
        )
        del self._queued[item.candidate.identity]
        return item.candidate

    def _start(
        self,
        identity: ExpertIdentity,
        *,
        demand: bool,
        deadline_ns: int | None,
    ) -> _Transfer | None:
        admission = self._cache.admit(identity)
        if admission.kind == AdmissionKind.RESIDENT:
            return None
        if admission.kind == AdmissionKind.INFLIGHT:
            transfer = self._transfers.get(identity)
            if transfer is None:
                raise RuntimeError("cache reports an unowned in-flight expert")
            transfer.demand |= demand
            return transfer
        submitted_ns = self._clock_ns()
        try:
            if admission.evicted is not None:
                self._on_evict(admission.evicted)
            future = self._storage.submit(identity)
        except BaseException:
            self._cache.fail(admission)
            raise
        transfer = _Transfer(
            admission=admission,
            future=future,
            submitted_ns=submitted_ns,
            predicted_completion_ns=submitted_ns + self._service_ns,
            deadline_ns=deadline_ns,
            demand=demand,
        )

        def mark_completed(_future) -> None:
            transfer.completed_ns = self._clock_ns()

        future.add_done_callback(mark_completed)
        self._transfers[identity] = transfer
        self.metrics["reads_submitted"] += 1
        if demand:
            self.metrics["demand_reads"] += 1
        return transfer

    def _pump(self, now_ns: int | None = None) -> None:
        now_ns = self._clock_ns() if now_ns is None else now_ns
        while len(self._transfers) < self._io_depth:
            candidate = self._next_candidate(now_ns)
            if candidate is None:
                break
            try:
                self._start(
                    candidate.identity,
                    demand=False,
                    deadline_ns=candidate.deadline_ns,
                )
            except ExpertCacheFullError:
                self.metrics["prefetch_dropped_capacity"] += 1

    def _begin_publication(self, transfer: _Transfer) -> None:
        transfer.record = transfer.future.result()
        transfer.publication = self._publisher.publish(
            transfer.record, transfer.admission.slot_id
        )

    def _abort_transfer(self, identity: ExpertIdentity, transfer: _Transfer) -> None:
        if transfer.record is not None:
            transfer.record.release()
        self._cache.fail(transfer.admission)
        self._transfers.pop(identity, None)
        self.metrics["publication_failures"] += 1

    def _finalize(self, identity: ExpertIdentity, transfer: _Transfer) -> None:
        assert transfer.publication is not None
        if not transfer.publication.release_if_ready():
            return
        self._on_resident(transfer.admission)
        self._cache.publish(transfer.admission)
        elapsed = max(
            (transfer.completed_ns or self._clock_ns()) - transfer.submitted_ns,
            1,
        )
        # A conservative rolling estimate reacts to tail growth but decays slowly.
        self._service_ns = max(elapsed, (7 * self._service_ns + elapsed) // 8)
        del self._transfers[identity]

    def poll(self) -> None:
        self._check_thread()
        now_ns = self._clock_ns()
        failures = []
        for identity, transfer in list(self._transfers.items()):
            if transfer.publication is None and transfer.future.done():
                if (
                    not transfer.demand
                    and transfer.deadline_ns is not None
                    and now_ns > transfer.deadline_ns
                ):
                    try:
                        record = transfer.future.result()
                        record.release()
                    finally:
                        self._cache.fail(transfer.admission)
                        del self._transfers[identity]
                        self.metrics["prefetch_expired"] += 1
                    continue
                try:
                    self._begin_publication(transfer)
                # Isolate one storage/publisher failure long enough to release
                # the other completed transfers, then fail the model thread.
                except Exception as exc:  # noqa: BLE001
                    self._abort_transfer(identity, transfer)
                    failures.append(exc)
                    continue
            if transfer.publication is not None:
                self._finalize(identity, transfer)
        self._pump()
        if failures:
            raise RuntimeError("expert publication failed") from failures[0]

    def _wait_and_publish(
        self,
        identity: ExpertIdentity,
        transfer: _Transfer,
        *,
        discard_expired: bool = False,
    ) -> bool:
        record = transfer.future.result()
        if (
            discard_expired
            and not transfer.demand
            and transfer.publication is None
            and transfer.deadline_ns is not None
            and self._clock_ns() > transfer.deadline_ns
        ):
            record.release()
            self._cache.fail(transfer.admission)
            del self._transfers[identity]
            self.metrics["prefetch_expired"] += 1
            return False
        if transfer.publication is None:
            self._begin_publication(transfer)
        assert transfer.publication is not None
        transfer.publication.wait()
        self._finalize(identity, transfer)
        return True

    def demand(self, identities: Iterable[ExpertIdentity]) -> list[SlotLease]:
        self._check_thread()
        requested = list(dict.fromkeys(identities))
        self.metrics["max_demand_batch_experts"] = max(
            self.metrics["max_demand_batch_experts"], len(requested)
        )
        if len(requested) > self._cache.capacity:
            raise ExpertCacheFullError(
                "demand batch requires "
                f"{len(requested)} experts but the partition has only "
                f"{self._cache.capacity} resident slots; increase the resident "
                "budget or split the MoE prefill working set"
            )
        leases: list[SlotLease] = []
        missing = deque[ExpertIdentity]()
        outstanding: dict[ExpertIdentity, _Transfer] = {}
        self.metrics["demand_batches"] += 1
        try:
            for identity in requested:
                self.metrics["demand_accesses"] += 1
                lease = self._cache.pin(identity)
                if lease is not None:
                    self.metrics["demand_cache_hits"] += 1
                    leases.append(lease)
                    continue
                self._queued.pop(identity, None)
                transfer = self._transfers.get(identity)
                if transfer is not None:
                    transfer.demand = True
                    self.metrics["reads_deduplicated"] += 1
                    outstanding[identity] = transfer
                else:
                    missing.append(identity)

            batch_misses = len(missing) + len(outstanding)
            self.metrics["demand_batch_misses"] += batch_misses
            self.metrics["max_demand_batch_misses"] = max(
                self.metrics["max_demand_batch_misses"], batch_misses
            )

            # Fill the configured storage queue before waiting.  The previous
            # implementation waited after every submit, reducing the demanded
            # path to effective QD=1 even when multiple experts in one routed
            # layer were cold.  Publishing and pinning remain owner-thread
            # operations, so batching I/O does not weaken slot lifetime rules.
            while missing or outstanding:
                while missing and len(self._transfers) < self._io_depth:
                    identity = missing.popleft()
                    transfer = self._start(identity, demand=True, deadline_ns=None)
                    if transfer is not None:
                        outstanding[identity] = transfer

                if not outstanding:
                    # The queue can be occupied entirely by unrelated
                    # prefetches.  Complete the oldest one without pumping more
                    # speculative work, then admit demanded experts first.
                    identity, transfer = min(
                        self._transfers.items(),
                        key=lambda item: item[1].submitted_ns,
                    )
                    try:
                        self._wait_and_publish(identity, transfer, discard_expired=True)
                    except BaseException:
                        if identity in self._transfers:
                            self._abort_transfer(identity, transfer)
                        raise
                    continue

                identity, transfer = min(
                    outstanding.items(),
                    key=lambda item: item[1].submitted_ns,
                )
                try:
                    self._wait_and_publish(identity, transfer)
                except BaseException:
                    if identity in self._transfers:
                        self._abort_transfer(identity, transfer)
                    raise
                del outstanding[identity]
                lease = self._cache.pin(identity)
                if lease is None:
                    raise RuntimeError("demanded expert was not published")
                leases.append(lease)

            self._pump()
            return leases
        except BaseException:
            for lease in leases:
                self._cache.unpin(lease)
            # Keep unrelated prefetches alive, but ensure every demanded read
            # already submitted by this batch is drained or failed so staging
            # buffers and LOADING slots cannot leak into the next request.
            for identity, transfer in list(outstanding.items()):
                if identity not in self._transfers:
                    continue
                try:
                    self._wait_and_publish(identity, transfer)
                except BaseException:
                    if identity in self._transfers:
                        self._abort_transfer(identity, transfer)
            self._pump()
            raise

    def release(self, leases: Iterable[SlotLease]) -> None:
        self._check_thread()
        for lease in leases:
            self._cache.unpin(lease)

    def snapshot_metrics(self) -> dict[str, int | float]:
        """Return low-volume owner-thread diagnostics without exposing state."""
        self._check_thread()
        accesses = self.metrics["demand_accesses"]
        return {
            **self.metrics,
            "demand_cache_hit_rate": (
                self.metrics["demand_cache_hits"] / accesses if accesses else 1.0
            ),
            "queued": len(self._queued),
            "transfers": len(self._transfers),
            "estimated_service_ms": self._service_ns / 1e6,
        }
