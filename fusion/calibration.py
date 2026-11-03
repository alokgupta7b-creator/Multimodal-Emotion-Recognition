"""Temperature scaling, used so that each model's confidence means something
before it is used as (part of) a reliability weight.

Temperature scaling on a probability vector ``p`` is ``softmax(log(p) / T)``.
``T > 1`` softens an over-confident model; ``T < 1`` sharpens an under-confident
one. ``T`` is fitted by minimising negative log-likelihood on a *development*
split -- never on the test split.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """Return ``softmax(log(P) / T)`` row-wise. ``P`` has shape ``(..., K)``."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"temperature must be positive and finite, got {T}")
    logits = np.log(np.clip(P, _EPS, 1.0)) / T
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def nll(P: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of integer labels ``y`` under rows of ``P``."""
    p = np.clip(P[np.arange(len(y)), y], _EPS, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    """Fit a single temperature by a coarse-to-fine grid search on NLL."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return 1.0
    grid = np.geomspace(0.05, 20.0, 120)
    losses = [nll(apply_temperature(P, t), y) for t in grid]
    best = grid[int(np.argmin(losses))]
    fine = np.geomspace(best / 1.15, best * 1.15, 60)
    losses = [nll(apply_temperature(P, t), y) for t in fine]
    return float(fine[int(np.argmin(losses))])


def expected_calibration_error(P: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE: weighted gap between confidence and accuracy per bin."""
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
