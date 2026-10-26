import random

import numpy as np
import pytest

from battleship_ai.agents import AGENTS, MCTSAgent, make_agent
from battleship_ai.game import Board, Observation, UNKNOWN
from battleship_ai.nn_model import WEIGHTS_PATH, _conv3x3


def play(agent, size, seed):
    board = Board.random(size, random.Random(seed))
    obs = Observation(size)
    while True:
        move = agent.select_move(obs)
        assert obs.grid[move] == UNKNOWN, "agent fired at a known cell"
        assert agent.insight and len(agent.insight["heatmap"]) == size
        result = board.fire(*move)
        obs.apply(result)
        if result.game_over:
            return len(obs.history)


needs_weights = pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="CNN weights not trained")


@pytest.mark.parametrize("size", [8, 12])
def test_random_agent_finishes(size):
    assert 17 <= play(make_agent("random", seed=1), size, 1) <= size * size


@pytest.mark.parametrize("size", [8, 10])
def test_mcts_agent_finishes_and_beats_random(size):
    agent = MCTSAgent(iterations=800, worlds=100, seed=2)
    shots = [play(agent, size, s) for s in range(3)]
    assert max(shots) < size * size * 0.85


@needs_weights
@pytest.mark.parametrize("size", [8, 10, 12])
def test_neural_agent_finishes_quickly(size):
    shots = [play(make_agent("neural", seed=3), size, s) for s in range(3)]
    assert sum(shots) / 3 < size * size * 0.6


def test_unknown_agent():
    with pytest.raises(ValueError):
        make_agent("nope")
    assert set(AGENTS) == {"random", "mcts", "neural"}


def test_numpy_conv_matches_naive_convolution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 5, 5)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((2, 4, 5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            ref[:, :, i, j] = np.einsum("bchw,ochw->bo", xp[:, :, i:i + 3, j:j + 3], w) + b
    np.testing.assert_allclose(_conv3x3(x, w, b), ref, rtol=1e-4, atol=1e-4)
