"""Input validation and normalisation for modality outputs.

Design rule: bad *model outputs* (NaN, negative, wrong sum, missing) never
crash the pipeline -- the offending modality is dropped and the reason is
reported as a status code. Bad *programmer input* (wrong number of classes,
unaligned rows, non-numeric data) raises ``FusionInputError``.

Status codes returned per (modality, sample):

``ok``           usable as given
``renormalized`` usable; did not sum to 1 within tolerance and was rescaled
``missing``      modality unavailable (``None`` or an all-NaN row)
``non_finite``   contains NaN/inf (but not entirely NaN), or its sum overflows -> dropped
``negative``     contains a negative value                -> dropped
``zero_sum``     all zeros                                -> dropped
``unsupported_mass``  real probability mass on a class the modality was declared
                 not to support (``FusionConfig.modality_support``). The data
                 contradicts the declaration, so the row is dropped instead of
                 having that mass silently discarded.             -> dropped
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .errors import FusionInputError

OK = "ok"
RENORMALIZED = "renormalized"
MISSING = "missing"
NON_FINITE = "non_finite"
NEGATIVE = "negative"
ZERO_SUM = "zero_sum"
UNSUPPORTED_MASS = "unsupported_mass"

USABLE = (OK, RENORMALIZED)
INVALID = (NON_FINITE, NEGATIVE, ZERO_SUM, UNSUPPORTED_MASS)

# Quality-score status codes.
Q_OK = "ok"
Q_UNKNOWN = "unknown"   # no score supplied (treated as 1.0 = no down-weighting)
Q_INVALID = "invalid"   # NaN/inf -> treated as unknown
Q_CLIPPED = "clipped"   # outside [0, 1] -> clipped


def validate_prob_rows(
    X: np.ndarray, n_classes: int, sum_tol: float = 1e-3, columns: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a matrix of probability rows.

    Parameters
    ----------
    X : array of shape ``(N, K)``. A row that is entirely NaN (over the
        supported columns) means "modality missing for this sample".
    n_classes : expected ``K``. A mismatch raises ``FusionInputError``.
    sum_tol : rows whose sum differs from 1 by more than this are flagged
        ``renormalized``.
    columns : optional boolean mask ``(K,)`` of the classes this model can
        predict. Only these columns are read; values elsewhere are ignored
        (they are *not* probabilities). Rows are normalised over the supported
        columns, and unsupported columns of ``clean`` are 0 -- a placeholder
        that downstream code must never interpret as a probability.

    Returns
    -------
    clean : ``(N, K)``; usable rows sum to 1 over the supported columns, invalid
        rows are uniform over them and must be ignored via ``valid``.
    valid : ``(N,)`` bool.
    reason : ``(N,)`` object array of status codes.
    """
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FusionInputError(f"probabilities are not numeric: {exc}") from None
    if X.ndim != 2 or X.shape[1] != n_classes:
        raise FusionInputError(f"expected probabilities of shape (N, {n_classes}), got {X.shape}")
    cols = np.ones(n_classes, dtype=bool) if columns is None else np.asarray(columns, dtype=bool)
    n = X.shape[0]
    Xs = X[:, cols]
    k = int(cols.sum())

    reason = np.full(n, OK, dtype=object)
    clean = np.zeros((n, n_classes))
    clean[:, cols] = 1.0 / k

    all_nan = np.isnan(Xs).all(axis=1)
    non_finite = ~np.isfinite(Xs).all(axis=1) & ~all_nan
    reason[all_nan] = MISSING
    reason[non_finite] = NON_FINITE

    ok = ~(all_nan | non_finite)
    if not cols.all():
        # NaN or (near-)zero placeholders are expected outside the support; anything else is a
        # contradiction between the declared support and the model output. The tolerance applies to
        # the TOTAL mass outside the support, not to each column (several small values add up).
        Xu = X[:, ~cols]
        with np.errstate(over="ignore", invalid="ignore"):
            outside = np.abs(np.where(np.isnan(Xu), 0.0, Xu)).sum(axis=1)
        unsupported = ok & ~(outside <= sum_tol)           # also catches inf
        reason[unsupported] = UNSUPPORTED_MASS
        ok &= ~unsupported
    safe = np.where(ok[:, None], Xs, 0.0)
    negative = ok & (safe < 0).any(axis=1)
    reason[negative] = NEGATIVE
    ok &= ~negative

    with np.errstate(over="ignore"):
        sums = np.where(ok, safe.sum(axis=1), 0.0)
    # Finite entries whose SUM overflows (e.g. 1e308 + 1e308) cannot be normalised: dividing by inf
    # would turn the row into all zeros that still counts as usable.
    overflow = ok & ~np.isfinite(sums)
    reason[overflow] = NON_FINITE
    ok &= ~overflow
    sums = np.where(ok, sums, 0.0)
    zero = ok & (sums <= 0)
    reason[zero] = ZERO_SUM
    ok &= ~zero

    off = ok & (np.abs(sums - 1.0) > sum_tol)
    reason[off] = RENORMALIZED

    sub = clean[:, cols]
    sub[ok] = safe[ok] / sums[ok, None]
    clean[:, cols] = sub
    return clean, ok, reason


def validate_quality_array(q, n: int, name: str = "modality") -> Tuple[np.ndarray, np.ndarray]:
    """Validate a per-sample quality score array.

    Returns ``(values, status)`` with values in [0, 1]. ``None`` -> all 1.0 with
    status ``unknown``; non-finite entries -> 1.0 with status ``invalid``;
    out-of-range entries are clipped with status ``clipped``.
    """
    if q is None:
        return np.ones(n), np.full(n, Q_UNKNOWN, dtype=object)
    try:
        arr = np.asarray(q, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FusionInputError(f"{name}: quality is not numeric: {exc}") from None
    if arr.shape != (n,):
        raise FusionInputError(f"{name}: quality must have shape ({n},), got {arr.shape}")
    status = np.full(n, Q_OK, dtype=object)
    bad = ~np.isfinite(arr)
    clipped = ~bad & ((arr < 0.0) | (arr > 1.0))
    status[bad] = Q_INVALID
    status[clipped] = Q_CLIPPED
    out = np.where(bad, 1.0, np.clip(arr, 0.0, 1.0))
    return out, status


def validate_quality(q: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    """Scalar convenience wrapper: ``(value_or_None, warning_or_None)``."""
    if q is None:
        return None, None
    try:
        vals, st = validate_quality_array(np.array([float(q)]), 1)
    except (TypeError, ValueError):
        return None, "quality is not a number; ignored"
    if st[0] == Q_INVALID:
        return None, "quality is not finite; ignored"
    if st[0] == Q_CLIPPED:
        return float(vals[0]), "quality outside [0, 1]; clipped"
    return float(vals[0]), None
