from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BlockPlacement:
    hot_blocks: tuple[int, ...]
    misses: tuple[tuple[int, int], ...]


class ActiveKVBlockDirectory:
    """Layer-local exact block directory with bounded global LRU storage.

    The directory intentionally contains no CUDA or storage code.  It owns the
    two-way logical/hot mapping and refuses to evict an incomplete write block.
    Keeping this policy independently testable is important: a stale mapping is
    silent model corruption, while an exhausted hot set is a recoverable error.
    """

    def __init__(self, *, num_layers: int, logical_blocks: int, hot_blocks: int):
        if min(num_layers, logical_blocks, hot_blocks) <= 0:
            raise ValueError("active-KV directory dimensions must be positive")
        self.num_layers = int(num_layers)
        self.logical_blocks = int(logical_blocks)
        self.hot_blocks = int(hot_blocks)
        self._logical_to_hot = [
            [-1] * self.logical_blocks for _ in range(self.num_layers)
        ]
        self._hot_to_logical = [
            [-1] * self.hot_blocks for _ in range(self.num_layers)
        ]
        self._last_used = [[0] * self.hot_blocks for _ in range(self.num_layers)]
        self._pinned = [[False] * self.hot_blocks for _ in range(self.num_layers)]
        self._authoritative = [
            [False] * self.logical_blocks for _ in range(self.num_layers)
        ]
        self._clock = 0

    def _validate(self, layer: int, logical_block: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"active-KV layer out of range: {layer}")
        if not 0 <= logical_block < self.logical_blocks:
            raise IndexError(f"active-KV logical block out of range: {logical_block}")

    def is_authoritative(self, layer: int, logical_block: int) -> bool:
        self._validate(layer, logical_block)
        return self._authoritative[layer][logical_block]

    def lookup(self, layer: int, logical_block: int) -> int:
        self._validate(layer, logical_block)
        return self._logical_to_hot[layer][logical_block]

    def place(
        self,
        layer: int,
        logical_blocks: Iterable[int],
        *,
        require_authoritative: bool,
    ) -> BlockPlacement:
        """Resolve blocks into hot slots, returning newly assigned misses.

        ``require_authoritative`` is true for reads.  A read of a block that has
        never completed write-through is rejected instead of returning stale or
        zero-filled KV.  Writes pass false and populate the assigned slot first.
        """

        unique = tuple(dict.fromkeys(int(block) for block in logical_blocks))
        for block in unique:
            self._validate(layer, block)
            if (
                require_authoritative
                and self._logical_to_hot[layer][block] < 0
                and not self._authoritative[layer][block]
            ):
                raise RuntimeError(
                    "active-KV read requested before authoritative write: "
                    f"layer={layer}, block={block}"
                )

        # Touch every currently resident member before assigning misses.  This
        # prevents a miss early in the compact GPU plan from evicting another
        # block selected by the same attention operation but listed later.
        resolved_by_block: dict[int, int] = {}
        misses: list[tuple[int, int]] = []
        for block in unique:
            hot = self._logical_to_hot[layer][block]
            if hot >= 0:
                self._clock += 1
                self._last_used[layer][hot] = self._clock
                resolved_by_block[block] = hot
        for block in unique:
            if block in resolved_by_block:
                continue
            hot = self._allocate_hot(layer)
            previous = self._hot_to_logical[layer][hot]
            if previous >= 0:
                self._logical_to_hot[layer][previous] = -1
            self._hot_to_logical[layer][hot] = block
            self._logical_to_hot[layer][block] = hot
            self._clock += 1
            self._last_used[layer][hot] = self._clock
            resolved_by_block[block] = hot
            misses.append((block, hot))
        return BlockPlacement(
            tuple(resolved_by_block[block] for block in unique), tuple(misses)
        )

    def _allocate_hot(self, layer: int) -> int:
        for hot, logical in enumerate(self._hot_to_logical[layer]):
            if logical < 0:
                return hot
        candidate = -1
        oldest = None
        for hot, used in enumerate(self._last_used[layer]):
            if self._pinned[layer][hot]:
                continue
            if oldest is None or used < oldest:
                candidate = hot
                oldest = used
        if candidate < 0:
            raise RuntimeError(
                "active-KV hot cache is exhausted by pinned partial blocks"
            )
        return candidate

    def begin_write(self, layer: int, logical_block: int, *, starts_block: bool) -> int:
        placement = self.place(
            layer, (logical_block,), require_authoritative=False
        )
        hot = placement.hot_blocks[0]
        if starts_block:
            # A recycled logical page starts a new generation.  It must not be
            # readable until all members have been written and persisted.
            self._authoritative[layer][logical_block] = False
        self._pinned[layer][hot] = True
        return hot

    def finish_write(self, layer: int, logical_block: int) -> None:
        self._validate(layer, logical_block)
        hot = self._logical_to_hot[layer][logical_block]
        if hot < 0:
            raise RuntimeError("completed active-KV block lost its hot placement")
        self._authoritative[layer][logical_block] = True
        self._pinned[layer][hot] = False

    def invalidate(self, layer: int, logical_blocks: Iterable[int]) -> None:
        for logical_block in logical_blocks:
            block = int(logical_block)
            self._validate(layer, block)
            hot = self._logical_to_hot[layer][block]
            if hot >= 0:
                self._logical_to_hot[layer][block] = -1
                self._hot_to_logical[layer][hot] = -1
                self._pinned[layer][hot] = False
            self._authoritative[layer][block] = False
