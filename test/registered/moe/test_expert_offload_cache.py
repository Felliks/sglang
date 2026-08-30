from __future__ import annotations

import pytest
from sglang.srt.layers.moe.expert_offload.cache import BoundedExpertCache
from sglang.srt.layers.moe.expert_offload.config import ExpertOffloadConfig
from sglang.srt.layers.moe.expert_offload.interfaces import (
    AdmissionKind,
    ExpertIdentity,
)
from sglang.srt.layers.moe.expert_offload.storage.memory import MemoryExpertStorage


def _load(cache: BoundedExpertCache, identity: ExpertIdentity) -> None:
    admission = cache.admit(identity)
    assert admission.kind == AdmissionKind.OWNER
    cache.publish(admission)


def test_generation_prevents_stale_publication() -> None:
    cache = BoundedExpertCache(capacity=1)
    first = cache.admit(ExpertIdentity(0, 1))
    cache.fail(first)
    second = cache.admit(ExpertIdentity(0, 2))

    with pytest.raises(RuntimeError, match="stale"):
        cache.publish(first)

    cache.publish(second)
    assert cache.resident_slot(ExpertIdentity(0, 2)) == 0


def test_pinned_slot_is_never_evicted() -> None:
    cache = BoundedExpertCache(capacity=1)
    identity = ExpertIdentity(0, 1)
    _load(cache, identity)
    lease = cache.pin(identity)
    assert lease is not None

    with pytest.raises(RuntimeError, match="no unpinned"):
        cache.admit(ExpertIdentity(0, 2))

    cache.unpin(lease)
    replacement = cache.admit(ExpertIdentity(0, 2))
    assert replacement.evicted == identity


def test_inflight_is_deduplicated_and_lru_is_causal() -> None:
    cache = BoundedExpertCache(capacity=2)
    first = ExpertIdentity(0, 1)
    second = ExpertIdentity(0, 2)
    third = ExpertIdentity(0, 3)
    admission = cache.admit(first)
    duplicate = cache.admit(first)
    assert duplicate.kind == AdmissionKind.INFLIGHT
    assert duplicate.generation == admission.generation
    cache.publish(admission)
    _load(cache, second)
    assert cache.resident_slot(first) == admission.slot_id

    replacement = cache.admit(third)
    assert replacement.evicted == second


def test_memory_storage_is_fixed_size_and_fail_closed() -> None:
    identity = ExpertIdentity(2, 7)
    storage = MemoryExpertStorage({identity: b"abcd"})
    destination = bytearray(4)
    storage.read_into(identity, memoryview(destination))
    assert destination == b"abcd"

    with pytest.raises(ValueError, match="expected 4"):
        storage.read_into(identity, memoryview(bytearray(3)))
    with pytest.raises(KeyError, match="unknown expert"):
        storage.read_into(ExpertIdentity(2, 8), memoryview(bytearray(4)))


def test_config_rejects_ambiguous_or_unbounded_storage() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ExpertOffloadConfig(backend="memory", resident_ratio=0.5, resident_gib=1)
    with pytest.raises(ValueError, match="storage_path"):
        ExpertOffloadConfig(backend="nvme", resident_ratio=0.5)

    config = ExpertOffloadConfig(backend="memory", resident_ratio=0.625)
    assert config.io_depth == 2
