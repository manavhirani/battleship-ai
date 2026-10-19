"""The deep-learned model: feature encoding and a pure-numpy forward pass.

Training (``battleship_ai.train``) uses PyTorch. Playing only needs numpy: the
trained weights are exported to ``weights/battleship_cnn.npz`` and evaluated
here with the same architecture.

Architecture: a fully convolutional residual network. It maps the observed
board to one logit per cell: the chance that an enemy ship is on that cell.
Because it is fully convolutional, one network plays every board size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .game import FLEET_NAMES, HIT, MISS, SUNK, UNKNOWN

WEIGHTS_PATH = Path(__file__).parent / "weights" / "battleship_cnn.npz"
N_FEATURES = 5 + len(FLEET_NAMES)


def encode(grids: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Features for a batch of observations.

    grids: (B, n, n) cell codes. alive: (B, 5) 1 if that fleet ship is afloat.
    Returns (B, 10, n, n) float32: unknown / miss / hit / sunk one-hot planes,
    a plane of ones (lets the zero padding mark the board edge), and one
    constant plane per fleet ship telling whether it is still afloat.
    """
    grids = np.asarray(grids)
    b, n, _ = grids.shape
    out = np.empty((b, N_FEATURES, n, n), dtype=np.float32)
    for k, code in enumerate((UNKNOWN, MISS, HIT, SUNK)):
        out[:, k] = grids == code
    out[:, 4] = 1.0
    out[:, 5:] = np.asarray(alive, dtype=np.float32)[:, :, None, None]
    return out


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """'Same' 3x3 convolution. x: (B, C, H, W), w: (O, C, 3, 3)."""
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # B,C,H,W,3,3
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(B * H * W, C * 9)
    y = cols @ w.reshape(w.shape[0], -1).T + b
    return y.reshape(B, H, W, -1).transpose(0, 3, 1, 2)


class NumpyCNN:
    def __init__(self, params: dict[str, np.ndarray], meta: dict) -> None:
        self.p = {k: v.astype(np.float32) for k, v in params.items()}
        self.meta = meta
        self.blocks = int(meta["blocks"])

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NumpyCNN":
        data = np.load(path or WEIGHTS_PATH)
        meta = json.loads(str(data["meta"]))
        return cls({k: data[k] for k in data.files if k != "meta"}, meta)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x: (B, 10, n, n) features -> (B, n, n) logits."""
        p = self.p
        h = np.maximum(_conv3x3(x, p["stem.w"], p["stem.b"]), 0)
        for i in range(self.blocks):
            y = np.maximum(_conv3x3(h, p[f"blocks.{i}.c1.w"], p[f"blocks.{i}.c1.b"]), 0)
            y = _conv3x3(y, p[f"blocks.{i}.c2.w"], p[f"blocks.{i}.c2.b"])
            h = np.maximum(h + y, 0)
        out = np.einsum("bchw,oc->bohw", h, p["head.w"][:, :, 0, 0]) + p["head.b"][None, :, None, None]
        return out[:, 0]

    def probabilities(self, grids: np.ndarray, alive: np.ndarray) -> np.ndarray:
        z = self.logits(encode(grids, alive))
        return 1.0 / (1.0 + np.exp(-z))


_cached: Optional[NumpyCNN] = None


def load_default() -> NumpyCNN:
    global _cached
    if _cached is None:
        _cached = NumpyCNN.load()
    return _cached
