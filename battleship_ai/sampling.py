"""Fast sampling of hidden ship layouts that agree with an Observation.

A layout is a Python int used as a bitmask: bit ``r * size + c`` is set when a
ship covers cell (r, c). Bitmask tests (``a & b``) are much faster than numpy
for the many small set operations that sampling and MCTS need.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Optional

import numpy as np

from .game import FLEET_SIZES, HIT, MISS, SUNK, UNKNOWN, Observation


@lru_cache(maxsize=None)
def placements(size: int, length: int) -> tuple[int, ...]:
    """Bitmasks of every position a ship of ``length`` can take on the board."""
    masks = []
    for r in range(size):
        for c in range(size - length + 1):
            masks.append(sum(1 << (r * size + c + i) for i in range(length)))
    for r in range(size - length + 1):
        for c in range(size):
            masks.append(sum(1 << ((r + i) * size + c) for i in range(length)))
    return tuple(masks)


@lru_cache(maxsize=None)
def covering(size: int, length: int) -> tuple[tuple[int, ...], ...]:
    """For each cell, the placements of a ship of ``length`` that cover it."""
    table: list[list[int]] = [[] for _ in range(size * size)]
    for mask in placements(size, length):
        m = mask
        while m:
            low = m & -m
            table[low.bit_length() - 1].append(mask)
            m ^= low
    return tuple(tuple(t) for t in table)


def grid_to_mask(grid: np.ndarray, *codes: int) -> int:
    flat = np.isin(grid.ravel(), codes)
    return sum(1 << int(i) for i in np.flatnonzero(flat))


def mask_to_grid(mask: int, size: int) -> np.ndarray:
    out = np.zeros(size * size, dtype=np.float32)
    while mask:
        low = mask & -mask
        out[low.bit_length() - 1] = 1.0
        mask ^= low
    return out.reshape(size, size)


class LayoutSampler:
    """Draws random layouts of the remaining enemy ships that are consistent
    with an observation:

    * no ship on a MISS or SUNK cell, and no overlap between ships;
    * every HIT cell (hit but not yet sunk) is covered by exactly one ship;
    * no remaining ship lies only on HIT cells (it would already be sunk).

    The sampler first covers the open hits, then places the other ships
    anywhere legal. It is fast but not exactly uniform over all consistent
    layouts, which is fine for Monte Carlo search.
    """

    def __init__(self, obs: Observation, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()
        self.size = obs.size
        self.lengths = [length for _, length in obs.remaining]
        self.blocked = grid_to_mask(obs.grid, MISS, SUNK)
        self.hits = grid_to_mask(obs.grid, HIT)
        self.unknown = grid_to_mask(obs.grid, UNKNOWN)
        # Placements that are legal on their own, per remaining length.
        self.free = {L: [m for m in placements(self.size, L)
                         if not m & self.blocked and m & self.unknown]
                     for L in set(self.lengths)}

    def sample(self) -> Optional[int]:
        """One consistent layout as a bitmask, or None if this attempt failed."""
        rng, blocked, hits = self.rng, self.blocked, self.hits
        todo = list(self.lengths)
        rng.shuffle(todo)
        occupied = 0
        uncovered = hits
        while uncovered:
            low = uncovered & -uncovered
            cell = low.bit_length() - 1
            options = []
            for i, L in enumerate(todo):
                if L in todo[:i]:
                    continue  # same length already listed: identical options
                for m in covering(self.size, L)[cell]:
                    if not m & (blocked | occupied) and m & self.unknown:
                        options.append((L, m))
            if not options:
                return None
            L, m = rng.choice(options)
            todo.remove(L)
            occupied |= m
            uncovered &= ~m
        busy = blocked | occupied | hits
        for L in todo:
            options = [m for m in self.free[L] if not m & busy]
            if not options:
                return None
            m = rng.choice(options)
            occupied |= m
            busy |= m
        return occupied

    def sample_many(self, n: int, max_attempts: Optional[int] = None) -> list[int]:
        max_attempts = max_attempts or n * 20
        out: list[int] = []
        for _ in range(max_attempts):
            layout = self.sample()
            if layout is not None:
                out.append(layout)
                if len(out) == n:
                    break
        return out


def random_layout_masks(size: int, rng: random.Random) -> list[int]:
    """A random full classic fleet, as one bitmask per ship (fleet order)."""
    while True:
        busy = 0
        ships = []
        for L in FLEET_SIZES:
            options = [m for m in placements(size, L) if not m & busy]
            if not options:
                break
            m = rng.choice(options)
            ships.append(m)
            busy |= m
        else:
            return ships
