from __future__ import annotations

import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertTensors,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput


class _Prepared:
    def __init__(self) -> None:
        slot_values = torch.zeros((2, 1, 1), dtype=torch.float32)
        slot_scales = torch.zeros((2, 1), dtype=torch.float32)
        self.tensors = MarlinExpertTensors(
            w13_qweight=slot_values,
            w2_qweight=slot_values,
            w13_scales=slot_scales,
            w2_scales=slot_scales,
            w13_global_scale=slot_scales,
            w2_global_scale=slot_scales,
        )
        self.expert_map = torch.full((512,), -1, dtype=torch.int32)
        self.expert_map[401] = 0
        self.expert_map[17] = 1
        self.completed = False

    def record_completion(self) -> None:
        self.completed = True


class _Offload:
    def __init__(self, prepared: _Prepared) -> None:
        self.prepared = prepared

    def prepare(self, topk_ids, *, hidden_states=None):
        assert topk_ids.tolist() == [[401, 17]]
        assert hidden_states is not None
        return self.prepared


def test_offloaded_global_ids_are_remapped_before_marlin_alignment(monkeypatch) -> None:
    prepared = _Prepared()
    captured = {}

    def fake_marlin_moe(**kwargs):
        captured.update(kwargs)
        return torch.zeros((1, 4), dtype=torch.bfloat16)

    monkeypatch.setattr(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe",
        fake_marlin_moe,
    )
    monkeypatch.setattr(
        "sglang.srt.layers.quantization.marlin_utils.marlin_make_workspace",
        lambda *_args, **_kwargs: torch.zeros(1, dtype=torch.int32),
    )

    hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
    topk = StandardTopKOutput(
        topk_weights=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
        topk_ids=torch.tensor([[401, 17]], dtype=torch.int32),
        router_logits=torch.zeros((1, 512), dtype=torch.float32),
    )
    dispatch = StandardDispatchOutput(hidden_states, None, topk)
    scales = torch.zeros((512, 1), dtype=torch.float32)
    quant_info = MarlinMoeQuantInfo(
        w13_qweight=torch.zeros((512, 1, 1)),
        w2_qweight=torch.zeros((512, 1, 1)),
        w13_scales=scales,
        w2_scales=scales,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        expert_offload=_Offload(prepared),
    )

    result = fused_experts_none_to_marlin(
        dispatch,
        quant_info,
        MoeRunnerConfig(activation="silu", is_gated=True),
    )

    assert result.hidden_states.shape == (1, 4)
    assert captured["topk_ids"].tolist() == [[0, 1]]
    assert captured["expert_map"] is None
    assert captured["global_num_experts"] == -1
    assert captured["w1"].shape[0] == 2
    assert prepared.completed
