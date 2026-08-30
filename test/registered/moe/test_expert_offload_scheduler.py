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
