"""Input validation and normalisation for modality outputs.

Design rule: bad *model outputs* (NaN, negative, wrong sum, missing) must never
crash the pipeline -- the offending modality is dropped and the reason is
reported. Bad *programmer input* (wrong number of classes, unknown modality
name) raises ``ValueError`` because it signals an interface mismatch that
should be fixed, not hidden.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Reason codes returned by ``validate_prob_rows``.
OK = "ok"
RENORMALIZED = "renormalized"  # valid, but did not sum to 1 within tolerance
MISSING = "missing"            # None / all-NaN row: modality unavailable
NON_FINITE = "non_finite"      # contains NaN/inf (but not entirely NaN)
NEGATIVE = "negative"
ZERO_SUM = "zero_sum"

USABLE = (OK, RENORMALIZED)


def validate_prob_rows(
    X: np.ndarray, n_classes: int, sum_tol: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a matrix of probability rows.

    Parameters
    ----------
    X : array of shape ``(N, K)``. A row that is entirely NaN means "modality
        missing for this sample".
    n_classes : expected ``K``. A mismatch raises ``ValueError``.
    sum_tol : rows whose sum differs from 1 by more than this are renormalised
        (and flagged ``renormalized``) instead of being trusted as-is.

    Returns
    -------
    clean : ``(N, K)`` float array, rows renormalised to sum to 1 (invalid rows
        are filled with a uniform distribution and must be ignored via ``valid``).
    valid : ``(N,)`` bool, True when the row is usable.
    reason : ``(N,)`` array of reason codes.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_classes:
        raise ValueError(
            f"expected probabilities of shape (N, {n_classes}), got {X.shape}"
        )
    n = X.shape[0]
    reason = np.full(n, OK, dtype=object)
    clean = np.full((n, n_classes), 1.0 / n_classes)

    all_nan = np.isnan(X).all(axis=1)
    non_finite = ~np.isfinite(X).all(axis=1) & ~all_nan
    reason[all_nan] = MISSING
    reason[non_finite] = NON_FINITE

    ok_so_far = ~(all_nan | non_finite)
    safe = np.where(ok_so_far[:, None], X, 0.0)
    negative = ok_so_far & (safe < 0).any(axis=1)
    reason[negative] = NEGATIVE

    ok_so_far &= ~negative
    sums = np.where(ok_so_far, safe.sum(axis=1), 0.0)
    zero = ok_so_far & (sums <= 0)
    reason[zero] = ZERO_SUM
    ok_so_far &= ~zero

    off = ok_so_far & (np.abs(sums - 1.0) > sum_tol)
    reason[off] = RENORMALIZED

    valid = ok_so_far
    clean[valid] = safe[valid] / sums[valid, None]
    return clean, valid, reason


def validate_quality(q: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    """Validate an external quality signal in [0, 1] (e.g. audio SNR score).

    Returns ``(value_or_None, warning_or_None)``. Non-finite values are ignored
    (treated as "unknown"); out-of-range values are clipped.
    """
    if q is None:
        return None, None
    try:
        qf = float(q)
    except (TypeError, ValueError):
        return None, "quality is not a number; ignored"
    if not np.isfinite(qf):
        return None, "quality is not finite; ignored"
    if qf < 0.0 or qf > 1.0:
        return float(np.clip(qf, 0.0, 1.0)), "quality outside [0, 1]; clipped"
    return qf, None
