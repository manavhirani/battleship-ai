import random

import pytest

from battleship_ai.game import (FLEET, HIT, MISS, SUNK, Board, IllegalMove, Observation, Ship)


@pytest.mark.parametrize("size", range(8, 13))
def test_random_board_is_a_legal_classic_fleet(size):
    board = Board.random(size, random.Random(size))
    assert sorted(s.name for s in board.ships) == sorted(n for n, _ in FLEET)
    cells = [cell for s in board.ships for cell in s.cells]
    assert len(cells) == len(set(cells)) == 17
    assert all(0 <= r < size and 0 <= c < size for r, c in cells)


def test_ships_can_touch_the_far_edges():
    # The old code had an off-by-one error that kept ships off the last row/column.
    seen = set()
    for seed in range(200):
        for s in Board.random(10, random.Random(seed)).ships:
            seen.update(s.cells)
    assert any(c == 9 for _, c in seen) and any(r == 9 for r, _ in seen)


def fleet(**overrides):
    ships = {"Carrier": Ship("Carrier", 5, 0, 0, True), "Battleship": Ship("Battleship", 4, 1, 0, True),
             "Cruiser": Ship("Cruiser", 3, 2, 0, True), "Submarine": Ship("Submarine", 3, 3, 0, True),
             "Destroyer": Ship("Destroyer", 2, 4, 0, True)}
    ships.update(overrides)
    return list(ships.values())


def test_invalid_fleets_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        Board(10, fleet(Destroyer=Ship("Destroyer", 2, 0, 3, False)))
    with pytest.raises(ValueError, match="outside"):
        Board(10, fleet(Destroyer=Ship("Destroyer", 2, 9, 9, True)))
    with pytest.raises(ValueError, match="length"):
        Board(10, fleet(Destroyer=Ship("Destroyer", 3, 4, 0, True)))
    with pytest.raises(ValueError, match="fleet"):
        Board(10, fleet()[:4])


def test_shots_hits_sinking_and_game_over():
    board = Board(10, fleet())
    obs = Observation(10)
    miss = board.fire(9, 9)
    assert not miss.hit and miss.label == "miss"
    obs.apply(miss)
    first = board.fire(4, 0)
    assert first.hit and first.sunk is None
    obs.apply(first)
    assert obs.grid[4, 0] == HIT
    sunk = board.fire(4, 1)
    assert sunk.sunk.name == "Destroyer" and sunk.label == "sunk"
    obs.apply(sunk)
    assert obs.grid[4, 0] == obs.grid[4, 1] == SUNK and obs.grid[9, 9] == MISS
    assert "Destroyer" not in [n for n, _ in obs.remaining]
    with pytest.raises(IllegalMove):
        board.fire(4, 1)
    with pytest.raises(IllegalMove):
        board.fire(10, 0)
    last = None
    for s in board.ships:
        for cell in s.cells:
            if cell not in board.shots:
                last = board.fire(*cell)
    assert last.game_over and board.all_sunk()
