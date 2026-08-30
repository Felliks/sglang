from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertOffloadAdapter,
    MarlinExpertTensors,
)


def _bundle(num_experts: int = 4) -> MarlinExpertTensors:
    values = torch.arange(num_experts, dtype=torch.float32).reshape(-1, 1)
    return MarlinExpertTensors(
        w13_qweight=values + 10,
        w2_qweight=values + 20,
        w13_scales=values + 30,
        w2_scales=values + 40,
        w13_global_scale=values + 50,
        w2_global_scale=values + 60,
    )


def test_adapter_copies_maps_and_pins_only_demanded_experts() -> None:
    adapter = MarlinExpertOffloadAdapter(
        layer_id=7, capacity=2, backing_tensors=_bundle()
    )
    prepared = adapter.prepare(torch.tensor([[3, 1, 3]], dtype=torch.int32))

    assert prepared.expert_map.tolist() == [-1, 0, -1, 1]
    assert prepared.tensors.w13_qweight[:, 0].tolist() == [11, 13]
    assert prepared.tensors.w2_global_scale[:, 0].tolist() == [61, 63]
    prepared.record_completion()


def test_adapter_uses_lru_slots_after_kernel_completion() -> None:
    adapter = MarlinExpertOffloadAdapter(
        layer_id=0, capacity=2, backing_tensors=_bundle()
    )
    first = adapter.prepare(torch.tensor([0, 1]))
    first.record_completion()
    touch = adapter.prepare(torch.tensor([0]))
    touch.record_completion()
    replacement = adapter.prepare(torch.tensor([2]))

    assert replacement.expert_map.tolist() == [0, -1, 1, -1]
    assert replacement.tensors.w13_qweight[:, 0].tolist() == [10, 12]
    replacement.record_completion()


def test_adapter_rejects_invalid_ids_and_ambiguous_completion() -> None:
    adapter = MarlinExpertOffloadAdapter(
        layer_id=0, capacity=2, backing_tensors=_bundle()
    )
    with pytest.raises(ValueError, match="outside"):
        adapter.prepare(torch.tensor([4]))
    with pytest.raises(RuntimeError, match="more experts"):
        adapter.prepare(torch.tensor([0, 1, 2]))

    prepared = adapter.prepare(torch.tensor([0]))
    prepared.record_completion()
    with pytest.raises(RuntimeError, match="already completed"):
        prepared.record_completion()
