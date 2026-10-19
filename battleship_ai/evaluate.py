"""Benchmark the agents.

The main metric is *shots to win*: how many shots an agent needs to sink a
random fleet (fewer is better). Head-to-head matches are decided by it too,
because players alternate shots.

    python -m battleship_ai evaluate --games 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .agents import AGENTS, make_agent
from .game import Board, Observation

RESULTS_PATH = Path(__file__).parent / "weights" / "benchmark.json"


def shots_to_win(agent_key: str, size: int, seed: int, **agent_kwargs) -> int:
    board = Board.random(size, random.Random(seed))
    agent = make_agent(agent_key, seed=seed + 1, **agent_kwargs)
    obs = Observation(size)
    while True:
        result = board.fire(*agent.select_move(obs))
        obs.apply(result)
        if result.game_over:
            return len(obs.history)


def _job(args):
    key, size, seed, kwargs = args
    return shots_to_win(key, size, seed, **kwargs)


def benchmark(keys, size: int, games: int, workers: int, seed: int = 1000, **kwargs) -> dict:
    out = {}
    with ProcessPoolExecutor(workers) as pool:
        for key in keys:
            t0 = time.perf_counter()
            # Same boards for every agent, so the comparison is fair.
            shots = list(pool.map(_job, [(key, size, seed + i, kwargs.get(key, {})) for i in range(games)]))
            out[key] = {
                "name": AGENTS[key].name,
                "games": games,
                "mean": round(statistics.mean(shots), 2),
                "stdev": round(statistics.pstdev(shots), 2),
                "median": statistics.median(shots),
                "best": min(shots),
                "worst": max(shots),
                "shots": shots,
                "seconds_per_game": round((time.perf_counter() - t0) * workers / games, 2),
            }
            print(f"{AGENTS[key].name:>10}: {out[key]['mean']:6.2f} shots to win "
                  f"(sd {out[key]['stdev']}, best {out[key]['best']}, worst {out[key]['worst']})")
    return out


def head_to_head(a: dict, b: dict) -> dict:
    """Win rates when a and b play the same boards; the first shooter is
    alternated, so a tie in shots goes to whoever shot first."""
    wins_a = wins_b = 0
    for i, (sa, sb) in enumerate(zip(a["shots"], b["shots"])):
        a_first = i % 2 == 0
        if sa < sb or (sa == sb and a_first):
            wins_a += 1
        else:
            wins_b += 1
    n = wins_a + wins_b
    return {"a": round(wins_a / n, 3), "b": round(wins_b / n, 3)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", default=list(AGENTS), choices=list(AGENTS))
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save", action="store_true", help=f"write results to {RESULTS_PATH}")
    args = ap.parse_args(argv)

    results = benchmark(args.agents, args.size, args.games, args.workers)
    matchups = {}
    keys = list(results)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            h = head_to_head(results[ka], results[kb])
            matchups[f"{ka}_vs_{kb}"] = h
            print(f"{results[ka]['name']} vs {results[kb]['name']}: "
                  f"{h['a'] * 100:.1f}% - {h['b'] * 100:.1f}%")
    if args.save:
        RESULTS_PATH.write_text(json.dumps({"size": args.size, "agents": results,
                                            "head_to_head": matchups}, indent=1))
        print(f"saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
