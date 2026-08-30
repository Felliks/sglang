from __future__ import annotations

from concurrent.futures import Future

from sglang.srt.layers.moe.expert_offload.cache import BoundedExpertCache
from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity
from sglang.srt.layers.moe.expert_offload.scheduler import (
    DeadlineExpertScheduler,
    PrefetchCandidate,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Record:
    def __init__(self, identity) -> None:
        self.identity = identity
        self.released = False

    def view(self):
        return memoryview(b"record")

    def release(self) -> None:
        self.released = True


class Storage:
    record_bytes = 6

    def __init__(self) -> None:
        self.futures = {}

    def submit(self, identity):
        future = Future()
        self.futures[identity] = future
        return future

    def complete(self, identity):
        self.futures[identity].set_result(Record(identity))

    def fail(self, identity):
        self.futures[identity].set_exception(OSError("read failed"))


class Publication:
    def __init__(self, record) -> None:
        self.record = record
        self.is_ready = True

    def ready(self):
        return self.is_ready

    def release_if_ready(self):
        if not self.is_ready:
            return False
        self.record.release()
        return True

    def wait(self):
        self.is_ready = True
        self.record.release()


class Publisher:
    def __init__(self) -> None:
        self.slots = {}

    def publish(self, record, slot_id):
        self.slots[record.identity] = slot_id
        return Publication(record)


class TrackingRecord(Record):
    def __init__(self, identity, storage) -> None:
        super().__init__(identity)
        self.storage = storage

    def release(self) -> None:
        if not self.released:
            self.storage.inflight -= 1
        super().release()


class TrackingStorage:
    """Immediately completes reads while retaining records until publication."""

    record_bytes = 6

    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0

    def submit(self, identity):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        future = Future()
        future.set_result(TrackingRecord(identity, self))
        return future


class OneFailureTrackingStorage(TrackingStorage):
    def __init__(self, failed_identity) -> None:
        super().__init__()
        self.failed_identity = failed_identity

    def submit(self, identity):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        future = Future()
        if identity == self.failed_identity:
            self.inflight -= 1
            future.set_exception(OSError("batched read failed"))
        else:
            future.set_result(TrackingRecord(identity, self))
        return future


def _scheduler(capacity=2, io_depth=2):
    clock = Clock()
    storage = Storage()
    publisher = Publisher()
    scheduler = DeadlineExpertScheduler(
        cache=BoundedExpertCache(capacity),
        storage=storage,
        publisher=publisher,
        io_depth=io_depth,
        initial_service_ns=100,
        clock_ns=clock,
    )
    return scheduler, storage, publisher, clock


def test_prefetch_is_deadline_aware_deduplicated_and_published() -> None:
    scheduler, storage, publisher, clock = _scheduler()
    first = ExpertIdentity(0, 1)
    late = ExpertIdentity(0, 2)
    scheduler.prefetch(
        [
            PrefetchCandidate(first, 0.9, 200),
            PrefetchCandidate(first, 0.8, 300),
            PrefetchCandidate(late, 0.9, 99),
        ]
    )
    assert scheduler.metrics["reads_submitted"] == 1
    assert scheduler.metrics["reads_deduplicated"] == 1
    assert scheduler.metrics["prefetch_rejected_deadline"] == 1
    storage.complete(first)
    clock.now = 80
    scheduler.poll()
    assert publisher.slots[first] in (0, 1)

    leases = scheduler.demand([first])
    assert len(leases) == 1
    scheduler.release(leases)


def test_demand_promotes_inflight_prefetch_and_blocks_for_correct_record() -> None:
    scheduler, storage, publisher, _ = _scheduler(capacity=1, io_depth=1)
    identity = ExpertIdentity(3, 7)
    scheduler.prefetch([PrefetchCandidate(identity, 1.0, 1000)])
    storage.complete(identity)
    leases = scheduler.demand([identity])

    assert publisher.slots[identity] == 0
    assert scheduler.metrics["reads_submitted"] == 1
    assert scheduler.metrics["reads_deduplicated"] == 1
    scheduler.release(leases)


def test_expired_completed_prefetch_is_discarded_not_published() -> None:
    scheduler, storage, publisher, clock = _scheduler(capacity=1, io_depth=1)
    identity = ExpertIdentity(0, 5)
    scheduler.prefetch([PrefetchCandidate(identity, 1.0, 150)])
    storage.complete(identity)
    clock.now = 151
    scheduler.poll()

    assert identity not in publisher.slots
    assert scheduler.metrics["prefetch_expired"] == 1


def test_failed_demand_releases_loading_slot_for_retry() -> None:
    scheduler, storage, _, _ = _scheduler(capacity=1, io_depth=1)
    failed = ExpertIdentity(0, 1)
    replacement = ExpertIdentity(0, 2)
    scheduler.prefetch([PrefetchCandidate(failed, 1.0, 1000)])
    storage.fail(failed)

    try:
        scheduler.demand([failed])
    except OSError as error:
        assert str(error) == "read failed"
    else:
        raise AssertionError("failed storage read did not propagate")

    scheduler.prefetch([PrefetchCandidate(replacement, 1.0, 1000)])
    assert replacement in storage.futures


def test_demand_fills_io_queue_before_waiting_for_publication() -> None:
    cache = BoundedExpertCache(4)
    storage = TrackingStorage()
    scheduler = DeadlineExpertScheduler(
        cache=cache,
        storage=storage,
        publisher=Publisher(),
        io_depth=2,
        initial_service_ns=100,
    )
    identities = [ExpertIdentity(0, expert_id) for expert_id in range(4)]

    leases = scheduler.demand(identities)

    assert storage.max_inflight == 2
    assert storage.inflight == 0
    assert len(leases) == 4
    assert scheduler.metrics["demand_batches"] == 1
    assert scheduler.metrics["demand_batch_misses"] == 4
    assert scheduler.metrics["max_demand_batch_misses"] == 4
    scheduler.release(leases)


def test_failed_batched_demand_releases_completed_leases_and_loading_slots() -> None:
    identities = [ExpertIdentity(0, expert_id) for expert_id in range(2)]
    cache = BoundedExpertCache(2)
    storage = OneFailureTrackingStorage(identities[1])
    scheduler = DeadlineExpertScheduler(
        cache=cache,
        storage=storage,
        publisher=Publisher(),
        io_depth=2,
        initial_service_ns=100,
    )

    try:
        scheduler.demand(identities)
    except OSError as error:
        assert str(error) == "batched read failed"
    else:
        raise AssertionError("failed batched storage read did not propagate")

    assert storage.max_inflight == 2
    assert storage.inflight == 0
    assert all(slot["pins"] == 0 for slot in cache.snapshot())
    assert all(slot["state"] != "loading" for slot in cache.snapshot())
