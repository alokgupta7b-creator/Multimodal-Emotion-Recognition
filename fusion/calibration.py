"""Temperature scaling, used so that each model's confidence means something
before it is used as (part of) a reliability weight.

Temperature scaling on a probability vector ``p`` is ``softmax(log(p) / T)``.
``T > 1`` softens an over-confident model; ``T < 1`` sharpens an under-confident
one. ``T`` is fitted by minimising negative log-likelihood on a *development*
split -- never on the test split.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-12
# Search range of ``fit_temperature``. A fitted value next to either bound means the data
# (nearly) prefers "no softening at all" or "completely flat" and should not be trusted.
T_MIN, T_MAX = 0.05, 20.0


def temperature_at_bound(T: float, margin: float = 1.25) -> bool:
    """True if ``T`` sits within ``margin`` x of the search range's edge."""
    return bool(T <= T_MIN * margin or T >= T_MAX / margin)


def apply_temperature(P: np.ndarray, T: float, log_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``softmax(log(P) / T)`` row-wise. ``P`` has shape ``(..., K)``.

    With ``log_prior`` (the log of the class prior the model's posterior assumes) only the EVIDENCE is
    scaled: ``z = log P - log_prior``; result ``softmax(z / T + log_prior)``. The prior is kept as it was,
    so dividing it out afterwards leaves exactly ``z / T`` (scaling ``log P`` as a whole would also
    scale the prior term, which later prior division cannot undo).
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"temperature must be positive and finite, got {T}")
    logits = np.log(np.clip(P, _EPS, 1.0))
    if log_prior is not None:
        logits = (logits - log_prior) / T + log_prior
    else:
        logits = logits / T
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def nll(P: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of integer labels ``y`` under rows of ``P``."""
    p = np.clip(P[np.arange(len(y)), y], _EPS, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(P: np.ndarray, y: np.ndarray, log_prior: Optional[np.ndarray] = None,
                    log_eval_prior: Optional[np.ndarray] = None) -> float:
    """Fit a single temperature by a coarse-to-fine grid search on NLL.

    Prior-aware form (``log_prior`` given): the evidence ``z = log P - log_prior`` is scaled by 1/T and
    scored as ``softmax(z / T + log_eval_prior)`` against ``y``. Pass the class prior of the labels ``y``
    as ``log_eval_prior`` (dev class frequencies), so a model trained under another prior is not
    mis-calibrated to absorb the prior shift between its training data and the dev set.
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return 1.0
    if log_prior is None:
        loss = lambda t: nll(apply_temperature(P, t), y)
    else:
        ev = log_prior if log_eval_prior is None else log_eval_prior
        z = np.log(np.clip(P, _EPS, 1.0)) - log_prior
        def loss(t):
            L = z / t + ev
            L = L - L.max(axis=1, keepdims=True)
            return float(-(L[np.arange(len(y)), y] - np.log(np.exp(L).sum(axis=1))).mean())
    grid = np.geomspace(T_MIN, T_MAX, 120)
    best = grid[int(np.argmin([loss(t) for t in grid]))]
    fine = np.geomspace(best / 1.15, best * 1.15, 60)
    return float(fine[int(np.argmin([loss(t) for t in fine]))])


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


def apply_temperature_masked(P: np.ndarray, T: float, mask: np.ndarray) -> np.ndarray:
    """Temperature-scale only the columns in ``mask`` (a model's supported classes).

    The supported columns are first renormalised to a conditional distribution
    (``P(c | c in support)``); other columns are returned as 0 -- a placeholder,
    not a probability.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(P.shape, dtype=float)
    sub = P[..., mask]
    sub = sub / np.clip(sub.sum(axis=-1, keepdims=True), _EPS, None)
    out[..., mask] = apply_temperature(sub, T)
    return out
