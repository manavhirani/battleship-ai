from __future__ import annotations

import numpy as np

from ..game import UNKNOWN, Observation
from .base import Agent


class RandomAgent(Agent):
    key = "random"
    name = "Random"
    tagline = "Fires blindly into the fog"
    description = ("Picks a uniformly random cell it has not fired at yet. It never "
                   "learns from hits, so it needs about 95 shots to win on a 10x10 board.")

    def select_move(self, obs: Observation) -> tuple[int, int]:
        moves = obs.legal_moves()
        self.insight = self.make_insight(
            "uniform", "Every open cell is equally likely",
            (obs.grid == UNKNOWN).astype(float), options=len(moves))
        return self.rng.choice(moves)
