"""Core Battleship rules: ships, boards, shots and what a shooter can observe.

Rules used everywhere in this project (engine, agents, training, UI):

* Square board, 8x8 to 12x12 (default 10x10).
* Classic fleet: Carrier 5, Battleship 4, Cruiser 3, Submarine 3, Destroyer 2.
* Ships are straight (horizontal or vertical) and do not overlap.
* Each turn a player fires one shot. The defender says "miss", "hit" or
  "sunk". When a ship is sunk, its name and position are revealed.
* The first player to sink the whole enemy fleet wins.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

FLEET: tuple[tuple[str, int], ...] = (
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
)
FLEET_NAMES = tuple(name for name, _ in FLEET)
FLEET_SIZES = tuple(size for _, size in FLEET)
MIN_SIZE, MAX_SIZE, DEFAULT_SIZE = 8, 12, 10

# Cell codes of an Observation grid (what the shooter knows about a cell).
UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3


class IllegalMove(ValueError):
    """Raised for a shot outside the board or at a cell already fired at."""


@dataclass(frozen=True)
class Ship:
    name: str
    length: int
    row: int
    col: int
    horizontal: bool

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        if self.horizontal:
            return tuple((self.row, self.col + i) for i in range(self.length))
        return tuple((self.row + i, self.col) for i in range(self.length))

    def to_dict(self) -> dict:
        return {"name": self.name, "length": self.length, "row": self.row,
                "col": self.col, "horizontal": self.horizontal}


@dataclass(frozen=True)
class ShotResult:
    row: int
    col: int
    hit: bool
    sunk: Optional[Ship] = None   # the ship this shot sank, if any
    game_over: bool = False

    @property
    def label(self) -> str:
        return "sunk" if self.sunk else ("hit" if self.hit else "miss")

    def to_dict(self) -> dict:
        return {"row": self.row, "col": self.col, "result": self.label,
                "sunk": self.sunk.to_dict() if self.sunk else None,
                "game_over": self.game_over}


def check_size(size: int) -> int:
    if not MIN_SIZE <= int(size) <= MAX_SIZE:
        raise ValueError(f"board size must be {MIN_SIZE}..{MAX_SIZE}, got {size}")
    return int(size)


class Board:
    """A player's own board: the hidden ship layout plus incoming shots."""

    def __init__(self, size: int, ships: Iterable[Ship]) -> None:
        self.size = check_size(size)
        self.ships: list[Ship] = list(ships)
        self._owner: dict[tuple[int, int], int] = {}
        self._validate()
        self.shots: set[tuple[int, int]] = set()
        self._damage = [0] * len(self.ships)

    def _validate(self) -> None:
        names = sorted(s.name for s in self.ships)
        if names != sorted(FLEET_NAMES):
            raise ValueError(f"fleet must be exactly {FLEET_NAMES}")
        for idx, ship in enumerate(self.ships):
            if ship.length != dict(FLEET)[ship.name]:
                raise ValueError(f"{ship.name} must have length {dict(FLEET)[ship.name]}")
            for cell in ship.cells:
                if not all(0 <= v < self.size for v in cell):
                    raise ValueError(f"{ship.name} is outside the board")
                if cell in self._owner:
                    raise ValueError(f"{ship.name} overlaps {self.ships[self._owner[cell]].name}")
                self._owner[cell] = idx

    @classmethod
    def random(cls, size: int = DEFAULT_SIZE, rng: Optional[random.Random] = None) -> "Board":
        rng = rng or random.Random()
        size = check_size(size)
        while True:
            taken: set[tuple[int, int]] = set()
            ships = []
            for name, length in FLEET:
                options = []
                for horizontal in (True, False):
                    rows = size if horizontal else size - length + 1
                    cols = size - length + 1 if horizontal else size
                    for r in range(rows):
                        for c in range(cols):
                            ship = Ship(name, length, r, c, horizontal)
                            if not taken.intersection(ship.cells):
                                options.append(ship)
                if not options:
                    break
                ship = rng.choice(options)
                taken.update(ship.cells)
                ships.append(ship)
            else:
                return cls(size, ships)

    @classmethod
    def from_dicts(cls, size: int, ships: Iterable[dict]) -> "Board":
        return cls(size, [Ship(str(s["name"]), dict(FLEET).get(str(s["name"]), -1), int(s["row"]),
                               int(s["col"]), bool(s["horizontal"])) for s in ships])

    def fire(self, row: int, col: int) -> ShotResult:
        cell = (int(row), int(col))
        if not all(0 <= v < self.size for v in cell):
            raise IllegalMove(f"{cell} is outside the board")
        if cell in self.shots:
            raise IllegalMove(f"{cell} was already fired at")
        self.shots.add(cell)
        idx = self._owner.get(cell)
        if idx is None:
            return ShotResult(*cell, hit=False)
        self._damage[idx] += 1
        sunk = self.ships[idx] if self._damage[idx] == self.ships[idx].length else None
        return ShotResult(*cell, hit=True, sunk=sunk, game_over=self.all_sunk())

    def all_sunk(self) -> bool:
        return all(d == s.length for d, s in zip(self._damage, self.ships))

    def is_sunk(self, ship: Ship) -> bool:
        return self._damage[self.ships.index(ship)] == ship.length

    def ship_mask(self) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=np.int8)
        for (r, c) in self._owner:
            mask[r, c] = 1
        return mask


class Observation:
    """Everything a shooter knows about the enemy board.

    This is the full input of every agent: the grid of UNKNOWN / MISS / HIT /
    SUNK codes and which enemy ships are still afloat.
    """

    def __init__(self, size: int) -> None:
        self.size = check_size(size)
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.sunk_ships: list[Ship] = []
        self.history: list[ShotResult] = []

    def apply(self, result: ShotResult) -> None:
        self.grid[result.row, result.col] = HIT if result.hit else MISS
        if result.sunk:
            self.sunk_ships.append(result.sunk)
            for r, c in result.sunk.cells:
                self.grid[r, c] = SUNK
        self.history.append(result)

    @property
    def remaining(self) -> list[tuple[str, int]]:
        """(name, length) of the enemy ships that are still afloat."""
        sunk = {s.name for s in self.sunk_ships}
        return [(n, l) for n, l in FLEET if n not in sunk]

    def legal_moves(self) -> list[tuple[int, int]]:
        return [tuple(map(int, rc)) for rc in np.argwhere(self.grid == UNKNOWN)]

    def copy(self) -> "Observation":
        other = Observation(self.size)
        other.grid = self.grid.copy()
        other.sunk_ships = list(self.sunk_ships)
        other.history = list(self.history)
        return other
