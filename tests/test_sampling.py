import random

import numpy as np

from battleship_ai.game import HIT, MISS, SUNK, Board, Observation
from battleship_ai.sampling import LayoutSampler, mask_to_grid, placements


def test_placement_counts():
    # A ship of length L fits in 2 * n * (n - L + 1) positions.
    assert len(placements(10, 5)) == 2 * 10 * 6
    assert len(placements(8, 2)) == 2 * 8 * 7


def test_samples_are_consistent_with_mid_game_observations():
    rng = random.Random(3)
    for game in range(25):
        size = rng.choice([8, 10, 12])
        board = Board.random(size, rng)
        obs = Observation(size)
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        for cell in cells[: rng.randint(5, size * size // 2)]:
            result = board.fire(*cell)
            obs.apply(result)
            if result.game_over:
                break
        if board.all_sunk():
            continue
        layouts = LayoutSampler(obs, random.Random(game)).sample_many(30)
        assert layouts, "sampler found no consistent layout"
        remaining_cells = sum(l for _, l in obs.remaining)
        for layout in layouts:
            grid = mask_to_grid(layout, size)
            assert grid.sum() == remaining_cells
            assert not grid[obs.grid == MISS].any()
            assert not grid[obs.grid == SUNK].any()
            assert grid[obs.grid == HIT].all()
