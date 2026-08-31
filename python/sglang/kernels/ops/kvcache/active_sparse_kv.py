from __future__ import annotations

import functools

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args


@functools.cache
def _jit_active_sparse_kv_module(
    block_tokens: int, record_bytes: int, block_size: int
):
    template_args = make_cpp_args(block_size, block_tokens, record_bytes)
    return load_jit(
        "active_sparse_kv",
        block_tokens,
        record_bytes,
        block_size,
        cuda_files=["active_sparse_kv.cuh"],
        cuda_wrappers=[
            ("unpack_qsa_records", f"unpack_qsa_records<{template_args}>"),
            ("pack_qsa_records", f"pack_qsa_records<{template_args}>"),
        ],
    )


def unpack_qsa_records(
    *,
    staging: torch.Tensor,
    destination_k: torch.Tensor,
    destination_v: torch.Tensor,
    destination_blocks: torch.Tensor,
    num_records: int,
    block_tokens: int = 4,
    block_size: int = 256,
) -> None:
    """Scatter packed pinned-host ``[K block | V block]`` records to CUDA."""

    module = _jit_active_sparse_kv_module(
        block_tokens, int(staging.shape[1]), block_size
    )
    module.unpack_qsa_records(
        staging,
        destination_k,
        destination_v,
        destination_blocks,
        num_records,
    )


def pack_qsa_records(
    *,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    source_blocks: torch.Tensor,
    staging: torch.Tensor,
    num_records: int,
    block_tokens: int = 4,
    block_size: int = 256,
) -> None:
    """Gather CUDA K/V blocks into packed pinned-host records for writeback."""

    module = _jit_active_sparse_kv_module(
        block_tokens, int(staging.shape[1]), block_size
    )
    module.pack_qsa_records(
        source_k,
        source_v,
        source_blocks,
        staging,
        num_records,
    )


__all__ = ["pack_qsa_records", "unpack_qsa_records"]
