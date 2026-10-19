from __future__ import annotations

from typing import Optional

from .base import Agent
from .mcts_agent import MCTSAgent
from .neural_agent import NeuralAgent
from .random_agent import RandomAgent

AGENTS: dict[str, type[Agent]] = {
    RandomAgent.key: RandomAgent,
    MCTSAgent.key: MCTSAgent,
    NeuralAgent.key: NeuralAgent,
}


def make_agent(key: str, seed: Optional[int] = None, **kwargs) -> Agent:
    try:
        return AGENTS[key](seed=seed, **kwargs)
    except KeyError:
        raise ValueError(f"unknown agent {key!r}; choose from {sorted(AGENTS)}") from None


__all__ = ["Agent", "AGENTS", "MCTSAgent", "NeuralAgent", "RandomAgent", "make_agent"]
