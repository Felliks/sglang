"""CPU tests for bounded per-token expert routing traces."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_distribution import (
    _SINGLE_PASS_GATHERER_KEY_PRIMARY,
    _SINGLE_PASS_GATHERER_KEY_SPECULATIVE_DRAFT,
    _DetailAccumulator,
    _DetailSinglePassGatherer,
)
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
    expert_distribution_recorder_buffer_size = 4
    expert_distribution_recorder_capture_speculative_draft = True

    def should_report_expert_balancedness(self):
        return False

    def get_model_config(self):
        return SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=10,
            ),
        )


class TestDetailSinglePassGatherer(CustomTestCase):
    def test_detail_accumulator_separates_draft_scope(self):
        accumulator = _DetailAccumulator(_ServerArgs(), _Metadata(), rank=0)
        self.assertEqual(
            accumulator.get_single_pass_gatherer_keys(),
            [
                _SINGLE_PASS_GATHERER_KEY_PRIMARY,
                _SINGLE_PASS_GATHERER_KEY_SPECULATIVE_DRAFT,
            ],
        )
        self.assertEqual(
            accumulator.get_single_pass_gatherer_key(
                _SINGLE_PASS_GATHERER_KEY_SPECULATIVE_DRAFT
            ),
            _SINGLE_PASS_GATHERER_KEY_SPECULATIVE_DRAFT,
        )

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
        self.assertEqual(output["active_layer_indices"], [0, 1, 2])
        self.assertEqual(output["router_input_token_indices"], [1, 2])
        expected = (torch.arange(48, dtype=torch.bfloat16).reshape(3, 16) + 2)[-2:]
        self.assertTrue(torch.equal(output["router_inputs_of_layer"][2], expected))

    def test_partial_draft_trace_only_copies_active_layers(self):
        forward_batch = SimpleNamespace(
            rids=["draft-0"],
            batch_size=1,
            input_ids=torch.tensor([7, 8]),
            positions=torch.tensor([11, 12]),
            extend_seq_lens_cpu=[2],
            forward_mode=SimpleNamespace(value="draft_extend_v2"),
        )
        with patch(
            "sglang.srt.eplb.expert_distribution.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=4),
        ):
            gatherer = _DetailSinglePassGatherer(_ServerArgs(), _Metadata(), rank=0)

        gatherer.reset()
        gatherer.on_forward_pass_start(forward_batch)
        gatherer.on_select_experts(
            0,
            torch.arange(20, dtype=torch.int32).reshape(2, 10) % 12,
            torch.arange(32, dtype=torch.bfloat16).reshape(2, 16),
            torch.arange(24, dtype=torch.bfloat16).reshape(2, 12),
        )

        output = gatherer.collect()
        self.assertEqual(output["active_layer_indices"], [0])
        self.assertEqual(output["router_inputs_of_layer"].shape, (1, 2, 16))
        self.assertEqual(output["router_logits_of_layer"].shape, (1, 2, 12))
        self.assertTrue(torch.all(output["topk_ids_of_layer"][1:] == -1))


if __name__ == "__main__":
    unittest.main()
