"""Choosing the abstention threshold (``FusionConfig.min_confidence``) from DEV data, without leakage.

The fused confidence (max fused probability) is a usable error signal, but the cut-off must come from data the
fuser was not fitted on: scoring dev rows with temperatures fitted on those same rows is optimistic. So
``out_of_fold_confidence`` splits dev by GROUP (dialogue id: utterances of one dialogue share speakers, scenes and
label patterns and must not straddle train and held-out folds), refits the fuser on the other folds and scores
the held-out fold. ``select_min_confidence`` then picks a threshold from those out-of-fold scores only.

This chooses a threshold; it does not change fusion. Nothing here is evidence about real performance until it is
run on real predictions.
"""
from __future__ import annotations

import warnings
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .config import FusionConfig
from .core import ReliabilityAwareFuser
from .errors import FusionInputError


def grouped_folds(groups, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per row; every group lands in exactly one fold (deterministic for a seed)."""
    g = np.asarray(groups)
    if g.ndim != 1:
        raise FusionInputError(f"groups must be 1-D, got shape {g.shape}")
    uniq = np.unique(g)
    if n_folds < 2 or n_folds > len(uniq):
        raise FusionInputError(f"n_folds must be in [2, number of groups = {len(uniq)}], got {n_folds}")
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {u: int(i % n_folds) for i, u in zip(range(len(uniq)), uniq[order])}
    return np.array([fold_of[v] for v in g], dtype=int)


def out_of_fold_confidence(config: FusionConfig, dev_probs: Mapping[str, np.ndarray], y_dev, groups,
                           n_folds: int = 5, seed: int = 0,
                           quality: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Fused confidence / prediction / correctness for every dev row, each scored by a fuser fitted WITHOUT its group.

    Returns arrays ``confidence``, ``pred``, ``correct`` (bool), ``fold``, ``conflict_score`` (mean pairwise JSD) and
    ``n_available`` (modalities used per row). Rows with no usable modality get
    ``pred = -1`` and are never counted as correct.
    """
    fold = grouped_folds(groups, n_folds, seed)
    from .core import _as_label_indices
    y = _as_label_indices(y_dev, config.labels)
    n = len(y)
    if len(fold) != n:
        raise FusionInputError(f"groups has {len(fold)} entries but y_dev has {n}")
    for m, pr in dev_probs.items():
        if pr is not None and np.asarray(pr).shape[0] != n:
            raise FusionInputError(f"{m}: dev probabilities have {np.asarray(pr).shape[0]} rows but y_dev has {n}")
    conf, pred, cscore = np.zeros(n), np.full(n, -1, dtype=int), np.zeros(n)
    n_avail = np.zeros(n, dtype=int)
    for k in range(n_folds):
        te, tr = fold == k, fold != k
        f = ReliabilityAwareFuser(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.fit({m: np.asarray(p)[tr] for m, p in dev_probs.items() if p is not None}, y[tr])
            q = None if quality is None else {m: np.asarray(v)[te] for m, v in quality.items() if v is not None}
            r = f.fuse_batch({m: np.asarray(p)[te] for m, p in dev_probs.items() if p is not None}, quality=q)
        conf[te], pred[te], cscore[te], n_avail[te] = r.confidence, r.pred, r.conflict_score, r.available.sum(axis=0)
    return {"confidence": conf, "pred": pred, "correct": (pred == y) & (pred >= 0), "fold": fold,
            "conflict_score": cscore, "n_available": n_avail}


def select_min_confidence(confidence, correct, coverage: Optional[float] = None,
                          max_risk: Optional[float] = None) -> Tuple[float, dict]:
    """Threshold for ``min_confidence`` from out-of-fold scores.

    Give exactly one of ``coverage`` (keep this fraction of predictions, most confident first) or ``max_risk``
    (largest error rate among kept predictions; picks the lowest threshold, i.e. the highest coverage, whose
    kept set has risk <= max_risk on these scores). Returns ``(threshold, info)``; ``info`` has the achieved
    coverage / risk on the supplied scores. If ``max_risk`` cannot be met the threshold is above every score
    (abstain on everything) and ``info['feasible']`` is False.
    """
    if (coverage is None) == (max_risk is None):
        raise FusionInputError("give exactly one of coverage or max_risk")
    c = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if c.shape != ok.shape or c.ndim != 1 or len(c) == 0:
        raise FusionInputError("confidence and correct must be non-empty 1-D arrays of equal length")
    order = np.argsort(-c, kind="stable")
    cs, errs = c[order], (~ok[order]).astype(float)
    kept = np.arange(1, len(c) + 1)
    risk = np.cumsum(errs) / kept                       # risk when keeping the top-k most confident
    if coverage is not None:
        if not 0.0 < coverage <= 1.0:
            raise FusionInputError("coverage must be in (0, 1]")
        k = max(1, int(np.ceil(coverage * len(c))))
        feasible = True
    else:
        if not 0.0 <= max_risk <= 1.0:
            raise FusionInputError("max_risk must be in [0, 1]")
        good = np.flatnonzero(risk <= max_risk)
        feasible = len(good) > 0
        k = int(kept[good[-1]]) if feasible else 0
    if k == 0:
        thr, cov, rk = float(np.nextafter(cs[0], np.inf)), 0.0, float("nan")
    else:
        thr = float(cs[k - 1])
        keep = c >= thr                                  # ties are kept together, so coverage may exceed the target
        cov, rk = float(keep.mean()), float((~ok[keep]).mean())
    return thr, {"threshold": thr, "coverage": cov, "risk": rk, "feasible": feasible,
                 "accuracy_all": float(ok.mean())}


def select_conflict_threshold(conflict_score, correct, n_available=None, max_flag_rate: float = 0.2,
                              min_lift: float = 1.2, z_crit: float = 1.645, min_flagged: int = 30
                              ) -> Tuple[Optional[float], dict]:
    """Dev-derived ``conflict_threshold``, or ``None`` when disagreement does not predict errors.

    Candidate rule: flag the rows with the highest disagreement (``conflict_score``), at most ``max_flag_rate`` of
    the rows that have two or more modalities. The flag is accepted only if, on these (out-of-fold) scores,
    flagged rows are wrong at least ``min_lift`` times as often as unflagged rows AND a one-sided two-proportion
    z-test reaches ``z_crit`` (1.645 = 5 %). Otherwise the flag carries no usable information and ``None`` is
    returned: leave the conflict flag off (``conflict_threshold=1.0``) and rely on confidence-based abstention.
    ``info`` always reports the flag rate, both risks, lift and z so the decision can be inspected.
    """
    sc = np.asarray(conflict_score, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if sc.shape != ok.shape or sc.ndim != 1:
        raise FusionInputError("conflict_score and correct must be 1-D arrays of equal length")
    if not 0.0 < max_flag_rate < 1.0:
        raise FusionInputError("max_flag_rate must be in (0, 1)")
    use = np.ones(len(sc), dtype=bool) if n_available is None else np.asarray(n_available) >= 2
    s, o = sc[use], ok[use]
    info = {"n_rows": int(use.sum()), "informative": False, "threshold": None}
    if len(s) < 2 * min_flagged:
        info["reason"] = f"only {len(s)} rows with >= 2 modalities (need {2 * min_flagged})"
        return None, info
    thr = float(np.quantile(s, 1.0 - max_flag_rate))
    flag = s >= thr
    n1, n0 = int(flag.sum()), int((~flag).sum())
    if n1 < min_flagged or n0 < min_flagged:
        info["reason"] = "too few flagged / unflagged rows (ties in conflict_score?)"
        return None, info
    r1, r0 = float((~o[flag]).mean()), float((~o[~flag]).mean())
    pool = float((~o).mean())
    se = np.sqrt(max(pool * (1 - pool), 1e-12) * (1.0 / n1 + 1.0 / n0))
    z = (r1 - r0) / se
    lift = r1 / r0 if r0 > 0 else float("inf")
    info.update(flag_rate=n1 / len(s), risk_flagged=r1, risk_unflagged=r0, lift=lift, z=float(z), candidate_threshold=thr)
    if lift >= min_lift and z >= z_crit:
        info.update(informative=True, threshold=thr)
        return thr, info
    info["reason"] = "disagreement does not predict errors on out-of-fold dev data"
    return None, info
