"""Information Set Monte Carlo Tree Search (ISMCTS) for Battleship.

Battleship hides the enemy layout, so a plain game tree is not available.
ISMCTS (Whitehouse, 2014) handles this with *determinization*:

1. Sample many hidden layouts that agree with everything observed so far
   (``LayoutSampler``).
2. Each iteration takes one sampled layout ("world") and runs one UCT pass:
   * selection   - descend the tree with UCB1 over shots;
   * expansion   - add one untried shot;
   * simulation  - play a short hunt/target rollout in that world;
   * backprop    - update every edge with the discounted number of hits.
   A shot's outcome (hit or miss) depends on the world, so each tree edge has
   one child per outcome.
3. Fire the most visited shot at the root.

The reward is the discounted hit count, so the search prefers shots that hit
soon and shots that set up quick follow-up hits.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from ..game import HIT, UNKNOWN, Observation
from ..sampling import LayoutSampler
from .base import Agent


class _Node:
    __slots__ = ("actions", "untried", "edges", "visits")

    def __init__(self, actions: tuple[int, ...], rng) -> None:
        self.actions = actions
        self.untried = list(actions)
        rng.shuffle(self.untried)
        self.edges: dict[int, list] = {}  # action -> [visits, value_sum, child_if_hit, child_if_miss]
        self.visits = 0


def _neighbours(size: int) -> list[tuple[int, ...]]:
    out = []
    for i in range(size * size):
        r, c = divmod(i, size)
        out.append(tuple(rr * size + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                         if 0 <= rr < size and 0 <= cc < size))
    return out


class MCTSAgent(Agent):
    key = "mcts"
    name = "MCTS"
    tagline = "Imagines thousands of possible fleets"
    description = ("Information Set Monte Carlo Tree Search. Each move it samples hundreds of "
                   "enemy layouts that fit what it has seen, then runs thousands of UCT "
                   "simulations across them and fires the most promising shot.")

    def __init__(self, iterations: int = 6000, worlds: int = 400, time_limit: float = 1.0,
                 exploration: float = 0.6, gamma: float = 0.85, horizon: int = 10,
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.iterations = iterations
        self.n_worlds = worlds
        self.time_limit = time_limit
        self.c = exploration
        self.gamma = gamma
        self.horizon = horizon

    def select_move(self, obs: Observation) -> tuple[int, int]:
        start = time.perf_counter()
        size, rng = obs.size, self.rng
        flat = obs.grid.ravel()
        unknown = tuple(int(i) for i in np.flatnonzero(flat == UNKNOWN))
        nbrs = _neighbours(size)
        # Rollouts first finish off ships that were hit but not sunk.
        seeds = [n for i in np.flatnonzero(flat == HIT) for n in nbrs[i] if flat[n] == UNKNOWN]

        worlds = LayoutSampler(obs, rng).sample_many(self.n_worlds)
        if not worlds:  # should not happen in a legal game; stay safe anyway
            move = rng.choice(seeds) if seeds else rng.choice(unknown)
            self.insight = self.make_insight("visits", "No consistent layout found",
                                             np.zeros((size, size)))
            return divmod(move, size)
        unknown_mask = sum(1 << i for i in unknown)
        targets = [bin(w & unknown_mask).count("1") for w in worlds]

        root = _Node(unknown, rng)
        gamma, c, horizon = self.gamma, self.c, self.horizon
        it = 0
        deadline = start + self.time_limit
        while it < self.iterations and (it & 63 or time.perf_counter() < deadline):
            w = it % len(worlds)
            world, target = worlds[w], targets[w]
            it += 1

            # Selection and expansion.
            node, path, rewards = root, [], []
            fired = hits = 0
            while True:
                expanded = False
                if node.untried:
                    a = node.untried.pop()
                    edge = node.edges[a] = [0, 0.0, None, None]
                    expanded = True
                else:
                    log_n = math.log(node.visits)
                    best, edge, a = -1.0, None, -1
                    for act, e in node.edges.items():
                        v = e[1] / e[0] + c * math.sqrt(log_n / e[0])
                        if v > best:
                            best, edge, a = v, e, act
                hit = (world >> a) & 1
                path.append((node, edge))
                rewards.append(hit)
                fired |= 1 << a
                hits += hit
                if expanded or hits == target or len(path) >= horizon:
                    break
                slot = 2 if hit else 3
                child = edge[slot]
                if child is None:
                    child = edge[slot] = _Node(tuple(x for x in node.actions if x != a), rng)
                node = child

            # Simulation: hunt at random, target the neighbours of hits.
            pending = [s for s in seeds if not (fired >> s) & 1]
            last = path[-1][0].actions
            while len(rewards) < horizon and hits < target:
                a = -1
                while pending:
                    cand = pending.pop()
                    if not (fired >> cand) & 1:
                        a = cand
                        break
                if a < 0:
                    for _ in range(8):
                        cand = last[int(rng.random() * len(last))]
                        if not (fired >> cand) & 1:
                            a = cand
                            break
                    else:
                        free = [x for x in last if not (fired >> x) & 1]
                        if not free:
                            break
                        a = rng.choice(free)
                hit = (world >> a) & 1
                fired |= 1 << a
                rewards.append(hit)
                hits += hit
                if hit:
                    pending.extend(nbrs[a])

            # Backpropagation of discounted returns.
            ret = 0.0
            returns = [0.0] * len(rewards)
            for t in range(len(rewards) - 1, -1, -1):
                ret = rewards[t] + gamma * ret
                returns[t] = ret
            for t, (n_, e_) in enumerate(path):
                n_.visits += 1
                e_[0] += 1
                e_[1] += returns[t]

        visits = np.zeros(size * size)
        for a, e in root.edges.items():
            visits[a] = e[0]
        move = int(max(root.edges, key=lambda a: (root.edges[a][0], root.edges[a][1])))
        e = root.edges[move]
        self.insight = self.make_insight(
            "visits", "Where MCTS spent its simulations",
            np.sqrt(visits.reshape(size, size)),  # sqrt keeps runner-up cells visible
            iterations=it, worlds=len(worlds),
            expected_return=round(e[1] / e[0], 3),
            hit_chance=round(sum((w >> move) & 1 for w in worlds) / len(worlds), 3),
            ms=round((time.perf_counter() - start) * 1000))
        return divmod(move, size)
