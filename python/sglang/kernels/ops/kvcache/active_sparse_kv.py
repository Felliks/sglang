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
            ("resolve_qsa_slots", f"resolve_qsa_slots<{template_args}>"),
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


def resolve_qsa_slots(
    *,
    logical_slots: torch.Tensor,
    logical_to_hot: torch.Tensor,
    hot_to_logical: torch.Tensor,
    physical_slots: torch.Tensor,
    selected_blocks: torch.Tensor,
    seen_epochs: torch.Tensor,
    selected_count: torch.Tensor,
    miss_count: torch.Tensor,
    epoch: int,
    block_tokens: int = 4,
    record_bytes: int = 8192,
    block_size: int = 256,
) -> None:
    """Resolve active-QSA slots and compact unique selected blocks on CUDA.

    The reverse directory is checked in the same kernel, so recycled hot slots
    cannot be mistaken for hits.  Only the compact block list ever needs to
    cross to CPU, and only when at least one miss requires an NVMe placement.
    """

    if logical_slots.dtype != torch.int32 or not logical_slots.is_contiguous():
        raise ValueError("logical_slots must be contiguous int32")
    module = _jit_active_sparse_kv_module(block_tokens, record_bytes, block_size)
    module.resolve_qsa_slots(
        logical_slots,
        logical_to_hot,
        hot_to_logical,
        physical_slots,
        selected_blocks,
        seen_epochs,
        selected_count,
        miss_count,
        epoch,
    )


__all__ = ["pack_qsa_records", "resolve_qsa_slots", "unpack_qsa_records"]
