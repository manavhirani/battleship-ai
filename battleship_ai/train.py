"""Train the deep-learned Battleship agent (needs PyTorch).

Method: supervised learning of ship occupancy on self-play positions.

* The network sees an observation (misses, hits, sunk ships, ships afloat)
  and predicts, per cell, whether a ship is there. The label is the true
  hidden layout, so the network learns P(ship | observation).
* Round 0 collects positions from random play. Each later round lets the
  current network play (with a little epsilon-random exploration), so it
  learns on the positions its own policy reaches (DAgger-style).
* Many games run in parallel as numpy arrays; boards of 8x8 to 12x12 are
  mixed so that the fully convolutional network plays every size.

    python -m battleship_ai train --rounds 14
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .game import FLEET_SIZES, HIT, MISS, SUNK, UNKNOWN
from .nn_model import N_FEATURES, WEIGHTS_PATH, NumpyCNN, encode
from .sampling import random_layout_masks

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    raise SystemExit("Training needs PyTorch: pip install -r requirements-train.txt")

SIZES = (8, 9, 10, 11, 12)
SIZE_WEIGHTS = (0.15, 0.15, 0.4, 0.15, 0.15)


class Block(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.c2.weight)  # each block starts as identity
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class BattleshipCNN(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 5) -> None:
        super().__init__()
        self.channels, self.n_blocks = channels, blocks
        self.stem = nn.Conv2d(N_FEATURES, channels, 3, padding=1)
        self.blocks = nn.ModuleList(Block(channels) for _ in range(blocks))
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)[:, 0]


def export(model: BattleshipCNN, path: Path, info: dict) -> None:
    rename = {"weight": "w", "bias": "b"}
    params = {}
    for key, value in model.state_dict().items():
        mod, kind = key.rsplit(".", 1)
        params[f"{mod}.{rename[kind]}"] = value.detach().cpu().numpy().astype(np.float16)
    meta = {"channels": model.channels, "blocks": model.n_blocks, **info}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **params)


# ---------------------------------------------------------------- self-play

def new_layouts(batch: int, size: int, rng: random.Random) -> np.ndarray:
    """(B, n, n) int8: 0 = water, k = ship k (1-based fleet index)."""
    ships = np.zeros((batch, size * size), dtype=np.int8)
    for b in range(batch):
        for k, mask in enumerate(random_layout_masks(size, rng), start=1):
            ships[b, [i for i in range(size * size) if mask >> i & 1]] = k
    return ships.reshape(batch, size, size)


@torch.no_grad()
def self_play(model, batch: int, size: int, epsilon: float, rng: random.Random,
              np_rng: np.random.Generator, keep: float = 1.0):
    """Play ``batch`` solo games at once. Returns recorded positions
    (grids, alive, targets) and the number of shots each game needed."""
    ships = new_layouts(batch, size, rng)
    grid = np.zeros_like(ships)
    alive = np.ones((batch, len(FLEET_SIZES)), dtype=np.int8)
    shots = np.zeros(batch, dtype=np.int32)
    done = np.zeros(batch, dtype=bool)
    rec_g, rec_a, rec_t = [], [], []
    while not done.all():
        act = np.flatnonzero(~done)
        g, a = grid[act], alive[act]
        if keep > 0:
            pick = np_rng.random(len(act)) < keep
            rec_g.append(g[pick])
            rec_a.append(a[pick])
            rec_t.append((ships[act][pick] > 0).astype(np.int8))
        rand = np_rng.random((len(act), size, size)).astype(np.float32)
        if model is None:
            scores = rand
        else:
            scores = model(torch.from_numpy(encode(g, a))).numpy()
            explore = np_rng.random(len(act)) < epsilon
            scores[explore] = rand[explore]
        scores[g != UNKNOWN] = -np.inf
        cell = scores.reshape(len(act), -1).argmax(1)
        r, c = cell // size, cell % size
        hit_id = ships[act, r, c]
        grid[act, r, c] = np.where(hit_id > 0, HIT, MISS)
        for i in np.flatnonzero(hit_id):
            b, k = act[i], hit_id[i]
            cells = ships[b] == k
            if (grid[b][cells] == HIT).all():
                grid[b][cells] = SUNK
                alive[b, k - 1] = 0
        shots[act] += 1
        done = alive.sum(1) == 0
    data = None
    if keep > 0:
        data = (np.concatenate(rec_g), np.concatenate(rec_a), np.concatenate(rec_t))
    return data, shots


def evaluate(model, games: int = 400, size: int = 10, seed: int = 7) -> float:
    model.eval()
    _, shots = self_play(model, games, size, 0.0, random.Random(seed),
                         np.random.default_rng(seed), keep=0)
    return float(shots.mean())


# ----------------------------------------------------------------- training

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=14)
    ap.add_argument("--games", type=int, default=2400, help="self-play games per round")
    ap.add_argument("--buffer-rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600, help="optimizer steps per round")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng, np_rng = random.Random(args.seed), np.random.default_rng(args.seed)
    model = BattleshipCNN(args.channels, args.blocks)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.rounds * args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    buffer: list[dict[int, tuple]] = []
    history = []
    t_start = time.time()

    for rnd in range(args.rounds):
        t0 = time.time()
        model.eval()
        round_data: dict[int, list] = {s: [] for s in SIZES}
        per_size = np_rng.multinomial(args.games, SIZE_WEIGHTS)
        policy = None if rnd == 0 else model
        for size, n_games in zip(SIZES, per_size):
            for start in range(0, n_games, 400):
                data, _ = self_play(policy, min(400, n_games - start), size, args.epsilon,
                                    rng, np_rng, keep=0.5)
                round_data[size].append(data)
        buffer.append({s: tuple(np.concatenate(x) for x in zip(*v)) for s, v in round_data.items() if v})
        buffer = buffer[-args.buffer_rounds:]
        pool = {s: tuple(np.concatenate(x) for x in zip(*[b[s] for b in buffer if s in b])) for s in SIZES}
        n_pos = sum(len(p[0]) for p in pool.values())
        t_play = time.time() - t0

        model.train()
        losses = []
        sizes = list(pool)
        weights = np.array([len(pool[s][0]) for s in sizes], dtype=float)
        for step in range(args.steps):
            size = sizes[np_rng.choice(len(sizes), p=weights / weights.sum())]
            g, a, t = pool[size]
            idx = np_rng.integers(0, len(g), args.batch)
            gb = g[idx]
            # Random flips/transposes: the rules are symmetric, so this is free data.
            if np_rng.random() < 0.5:
                gb, tb = gb[:, ::-1], t[idx][:, ::-1]
            else:
                tb = t[idx]
            if np_rng.random() < 0.5:
                gb, tb = gb[:, :, ::-1], tb[:, :, ::-1]
            if np_rng.random() < 0.5:
                gb, tb = gb.transpose(0, 2, 1), tb.transpose(0, 2, 1)
            x = torch.from_numpy(encode(np.ascontiguousarray(gb), a[idx]))
            y = torch.from_numpy(np.ascontiguousarray(tb)).float()
            mask = torch.from_numpy(np.ascontiguousarray(gb) == UNKNOWN).float()
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        score = evaluate(model)
        history.append({"round": rnd, "loss": round(float(np.mean(losses[-100:])), 4),
                        "shots_10x10": round(score, 2), "positions": n_pos})
        print(f"round {rnd:2d} | positions {n_pos:7d} | loss {history[-1]['loss']:.4f} | "
              f"greedy shots-to-win 10x10: {score:5.2f} | play {t_play:4.0f}s "
              f"train {time.time() - t0 - t_play:4.0f}s", flush=True)
        export(model, args.out, {"history": history, "train_seconds": round(time.time() - t_start)})

    # Check that the numpy forward pass matches PyTorch.
    model.eval()
    g, a, _ = pool[10]
    x = encode(g[:64], a[:64])
    ref = model(torch.from_numpy(x)).detach().numpy()
    got = NumpyCNN.load(args.out).logits(x)
    print(f"numpy vs torch max abs diff: {np.abs(ref - got).max():.4f} (fp16 weights)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
