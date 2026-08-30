"""CPU tests for bounded per-token expert routing traces."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_distribution import _DetailSinglePassGatherer
from sglang.test.test_utils import CustomTestCase


class _Metadata:
    num_layers = 3
    num_physical_experts = 12
    num_logical_experts = 12
    ep_size = 1


class _ServerArgs:
    device = "cpu"
    enable_two_batch_overlap = False
    expert_distribution_recorder_capture_router_inputs = True
    expert_distribution_recorder_max_router_input_tokens_per_pass = 2

    def get_model_config(self):
        return SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=10,
            ),
        )


class TestDetailSinglePassGatherer(CustomTestCase):
    def test_model_topk_and_bounded_router_inputs(self):
        forward_batch = SimpleNamespace(
            rids=["request-0"],
            batch_size=1,
            input_ids=torch.tensor([1, 2, 3]),
            positions=torch.tensor([7, 8, 9]),
            extend_seq_lens_cpu=[3],
            forward_mode=SimpleNamespace(value="decode"),
        )
        with patch(
            "sglang.srt.eplb.expert_distribution.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=4),
        ):
            gatherer = _DetailSinglePassGatherer(_ServerArgs(), _Metadata(), rank=0)

        gatherer.reset()
        gatherer.on_forward_pass_start(forward_batch)
        for layer_idx in range(_Metadata.num_layers):
            hidden_states = (
                torch.arange(48, dtype=torch.bfloat16).reshape(3, 16) + layer_idx
            )
            router_logits = (
                torch.arange(36, dtype=torch.bfloat16).reshape(3, 12) + layer_idx
            )
            topk_ids = torch.arange(30, dtype=torch.int32).reshape(3, 10) % 12
            gatherer.on_select_experts(
                layer_idx, topk_ids, hidden_states, router_logits
            )

        output = gatherer.collect()
        self.assertEqual(output["topk_ids_of_layer"].shape, (3, 3, 10))
        self.assertEqual(output["router_inputs_of_layer"].shape, (3, 2, 16))
        self.assertEqual(output["router_logits_of_layer"].shape, (3, 2, 12))
        self.assertEqual(output["router_input_token_indices"], [1, 2])
        expected = (torch.arange(48, dtype=torch.bfloat16).reshape(3, 16) + 2)[-2:]
        self.assertTrue(torch.equal(output["router_inputs_of_layer"][2], expected))


if __name__ == "__main__":
    unittest.main()
