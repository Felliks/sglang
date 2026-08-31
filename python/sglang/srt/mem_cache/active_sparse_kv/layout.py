from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveSparseKVLayout:
    """Fixed-extent layout for exact block-granular K+V records.

    Records are layer-major.  A logical full-KV slot is first reduced to its
    compression block, preserving SGLang's page/radix sharing because all
    tokens in one compression group map to the same immutable record slot.
    """

    num_layers: int
    logical_token_capacity: int
    page_size: int
    block_tokens: int
    record_bytes: int
    alignment: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "logical_token_capacity",
            "page_size",
            "block_tokens",
            "record_bytes",
            "alignment",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.page_size % self.block_tokens:
            raise ValueError("page_size must be divisible by block_tokens")
        if self.record_bytes % self.alignment:
            raise ValueError("record_bytes must be aligned for O_DIRECT")

    @property
    def block_capacity(self) -> int:
        # Include the allocator's reserved padding page, matching KV pools.
        token_slots = self.logical_token_capacity + self.page_size
        return (token_slots + self.block_tokens - 1) // self.block_tokens

    @property
    def layer_bytes(self) -> int:
        return self.block_capacity * self.record_bytes

    @property
    def file_bytes(self) -> int:
        return self.num_layers * self.layer_bytes

    def block_slot(self, logical_token_slot: int) -> int:
        if not 0 <= logical_token_slot < self.logical_token_capacity + self.page_size:
            raise IndexError(f"logical token slot out of range: {logical_token_slot}")
        return logical_token_slot // self.block_tokens

    def record_offset(self, layer_offset: int, block_slot: int) -> int:
        if not 0 <= layer_offset < self.num_layers:
            raise IndexError(f"layer offset out of range: {layer_offset}")
        if not 0 <= block_slot < self.block_capacity:
            raise IndexError(f"block slot out of range: {block_slot}")
        return layer_offset * self.layer_bytes + block_slot * self.record_bytes

    def validate_capacity(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.file_bytes > max_bytes:
            raise ValueError(
                "active sparse-KV extent exceeds its configured bound: "
                f"required={self.file_bytes}, max={max_bytes}"
            )
