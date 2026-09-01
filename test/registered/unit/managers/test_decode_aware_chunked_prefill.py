import pytest

from sglang.srt.managers.scheduler import (
    _decode_aware_chunked_prefill_size,
    _should_prioritize_decode,
)
from sglang.srt.server_args import ServerArgs


def test_idle_prefill_uses_throughput_ceiling():
    assert _decode_aware_chunked_prefill_size(2048, 128, 0) == 2048


def test_live_decode_uses_latency_budget():
    assert _decode_aware_chunked_prefill_size(2048, 128, 1) == 128
    assert _decode_aware_chunked_prefill_size(2048, 128, 4) == 128


def test_unconfigured_policy_preserves_existing_behavior():
    assert _decode_aware_chunked_prefill_size(2048, None, 4) == 2048
    assert _decode_aware_chunked_prefill_size(None, 128, 4) is None


def test_decode_priority_quota_only_applies_to_contended_prefill():
    assert _should_prioritize_decode(8, 8, True, True)
    assert not _should_prioritize_decode(0, 8, True, True)
    assert not _should_prioritize_decode(8, None, True, True)
    assert not _should_prioritize_decode(8, 8, False, True)
    assert not _should_prioritize_decode(8, 8, True, False)


def test_decode_budget_must_fit_the_allocated_chunk():
    valid = ServerArgs(
        model_path="dummy",
        served_model_name="dummy",
        chunked_prefill_size=2048,
        chunked_prefill_size_when_decode=128,
        page_size=64,
    )
    valid.check_server_args()

    invalid = ServerArgs(
        model_path="dummy",
        served_model_name="dummy",
        chunked_prefill_size=128,
        chunked_prefill_size_when_decode=256,
        page_size=64,
    )
    with pytest.raises(AssertionError, match="must not exceed"):
        invalid.check_server_args()


def test_decode_steps_per_prefill_validation():
    valid = ServerArgs(
        model_path="dummy",
        served_model_name="dummy",
        chunked_prefill_size=2048,
        decode_steps_per_prefill=8,
        page_size=64,
    )
    valid.check_server_args()

    invalid = ServerArgs(
        model_path="dummy",
        served_model_name="dummy",
        chunked_prefill_size=2048,
        decode_steps_per_prefill=0,
        page_size=64,
    )
    with pytest.raises(AssertionError, match="must be positive"):
        invalid.check_server_args()
