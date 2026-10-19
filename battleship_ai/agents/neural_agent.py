from __future__ import annotations

from typing import Optional

import numpy as np

from ..game import FLEET_NAMES, UNKNOWN, Observation
from ..nn_model import NumpyCNN, load_default
from .base import Agent


class NeuralAgent(Agent):
    key = "neural"
    name = "Deep CNN"
    tagline = "A residual network trained on millions of positions"
    description = ("A fully convolutional residual network that reads the board and predicts, "
                   "for every cell, the chance that a ship is there. It learned this from "
                   "self-play games, then fires at the most likely cell.")

    def __init__(self, model: Optional[NumpyCNN] = None, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.model = model or load_default()

    def select_move(self, obs: Observation) -> tuple[int, int]:
        alive_names = {name for name, _ in obs.remaining}
        alive = np.array([[name in alive_names for name in FLEET_NAMES]], dtype=np.float32)
        prob = self.model.probabilities(obs.grid[None], alive)[0]
        prob = np.where(obs.grid == UNKNOWN, prob, 0.0)
        best = np.flatnonzero(prob == prob.max())
        r, c = divmod(int(self.rng.choice(list(best))), obs.size)
        self.insight = self.make_insight(
            "probability", "Predicted chance of a ship on each cell", prob,
            hit_chance=round(float(prob[r, c]), 3),
            expected_ship_cells=round(float(prob.sum()), 1))
        return r, c
