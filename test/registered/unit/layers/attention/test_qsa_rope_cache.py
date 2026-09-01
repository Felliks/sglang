from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import (
    _ensure_rope_cache_covers_positions,
)


class _PositionsWithoutHostRead:
    def max(self):
        raise AssertionError("a complete static RoPE cache must not read CUDA positions")


def test_complete_static_rope_cache_avoids_position_host_read():
    rotary_emb = Mock()
    rotary_emb.max_position_embeddings = 16
    rotary_emb.cos_sin_cache = torch.empty(16, 8)

    _ensure_rope_cache_covers_positions(rotary_emb, _PositionsWithoutHostRead())

    rotary_emb._ensure_cos_sin_cache_length.assert_not_called()


def test_partial_rope_cache_preserves_dynamic_growth():
    rotary_emb = Mock()
    rotary_emb.max_position_embeddings = 16
    rotary_emb.cos_sin_cache = torch.empty(8, 8)

    _ensure_rope_cache_covers_positions(rotary_emb, torch.tensor([2, 11, 7]))

    rotary_emb._ensure_cos_sin_cache_length.assert_called_once_with(11)
