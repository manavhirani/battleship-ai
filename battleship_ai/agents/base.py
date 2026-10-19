from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..game import Observation


class Agent:
    """An agent picks a shot from what it can observe of the enemy board.

    After ``select_move`` the agent sets ``insight``: a heatmap of how much it
    favours each cell plus some stats. The web UI draws it as "the AI's mind".
    """

    key = "base"
    name = "Agent"
    tagline = ""
    description = ""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.insight: Optional[dict] = None

    def select_move(self, obs: Observation) -> tuple[int, int]:
        raise NotImplementedError

    @staticmethod
    def make_insight(kind: str, label: str, heatmap: np.ndarray, **stats) -> dict:
        heat = np.asarray(heatmap, dtype=float)
        top = heat.max()
        if top > 0:
            heat = heat / top
        return {"kind": kind, "label": label,
                "heatmap": np.round(heat, 3).tolist(), "stats": stats}
