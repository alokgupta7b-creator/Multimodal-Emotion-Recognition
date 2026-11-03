"""Decision-level fusion of text / audio / face emotion probabilities.

Public API
----------
``equal_weight_fusion(inputs)``           reference baseline (plain average)
``ReliabilityAwareFuser``                 calibrated, reliability-weighted fusion
    ``.fit(dev_probs, y_dev)``            fit temperatures + modality priors on a DEV split
    ``.fuse(inputs)``                     one sample  -> ``FusionResult``
    ``.fuse_batch(probs, quality)``       many samples -> ``BatchResult`` (vectorised)

Input for one sample is a mapping ``{"text": ..., "audio": ..., "face": ...}``
where each value is a probability vector over ``labels`` (list/array),
a ``ModalityInput`` (adds an optional ``quality`` in [0, 1]), or ``None``
when that modality is unavailable. Omitted modalities count as unavailable.

Reliability weight of modality ``m`` on one sample::

    w_m = prior_m * conf_m**gamma * quality_m**beta * consistency_m**lambda

* ``prior_m``       dev-set macro-F1 of the modality (how good it is in general)
* ``conf_m``        1 - normalised entropy of its *calibrated* prediction
* ``quality_m``     external signal in [0, 1] (audio SNR score, face-detected
                    ratio, ...); 1.0 when unknown
* ``consistency_m`` 1 - mean Jensen-Shannon divergence to the other available
                    modalities (agreement with the rest)

Each factor can be switched off by setting its exponent to 0. Weights are
renormalised over the *available* modalities only, so a missing modality is
never treated as a "neutral" vote.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .calibration import apply_temperature, fit_temperature
from .labels import MELD_LABELS, MODALITIES
from .metrics import macro_f1
from .validation import USABLE, validate_prob_rows, validate_quality

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModalityInput:
    """One modality's output for one sample."""

    probs: Optional[Sequence[float]] = None  # None => modality unavailable
    quality: Optional[float] = None          # optional signal in [0, 1]


InputLike = Union[ModalityInput, Sequence[float], np.ndarray, None]


@dataclass
class FusionResult:
    """Final prediction for one sample plus everything needed to explain it."""

    label: Optional[str]            # None when no usable modality existed
    label_index: int                # -1 when label is None
    probs: np.ndarray               # fused distribution over labels
    confidence: float               # max fused probability (0.0 if no prediction)
    margin: float                   # top-1 minus top-2 fused probability
    available: List[str]            # modalities that contributed
    missing: List[str]              # modalities absent or dropped as invalid
    weights: Dict[str, float]       # normalised weights (sum to 1 over available)
    conflict_score: float           # mean pairwise JSD in [0, 1] among available
    conflict: bool                  # modalities disagree (see conflict_threshold)
    uncertain: bool                 # conflict, or confidence < min_confidence
    method: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["probs"] = [float(x) for x in self.probs]
        return d


@dataclass
class BatchResult:
    probs: np.ndarray               # (N, K)
    pred: np.ndarray                # (N,) int, -1 = no prediction
    confidence: np.ndarray          # (N,)
    weights: np.ndarray             # (M, N) normalised
    available: np.ndarray           # (M, N) bool
    conflict_score: np.ndarray      # (N,)
    conflict: np.ndarray            # (N,) bool
    uncertain: np.ndarray           # (N,) bool
    modalities: tuple


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coerce(x: InputLike) -> ModalityInput:
    if isinstance(x, ModalityInput):
        return x
    return ModalityInput(probs=x)


def _entropy_norm(P: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy in [0, 1] along the last axis."""
    K = P.shape[-1]
    ent = -(P * np.log(np.clip(P, _EPS, 1.0))).sum(axis=-1)
    return np.clip(ent / np.log(K), 0.0, 1.0)


def _jsd(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits (range [0, 1]) along the last axis."""
    M = 0.5 * (P + Q)

    def kl(A, B):
        return (A * (np.log2(np.clip(A, _EPS, 1.0)) - np.log2(np.clip(B, _EPS, 1.0)))).sum(axis=-1)

    return np.clip(0.5 * kl(P, M) + 0.5 * kl(Q, M), 0.0, 1.0)


def _softmax_rows(L: np.ndarray) -> np.ndarray:
    L = L - L.max(axis=-1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(axis=-1, keepdims=True)


def _stack(
    probs: Mapping[str, np.ndarray],
    quality: Optional[Mapping[str, Optional[np.ndarray]]],
    modalities: Sequence[str],
    K: int,
    sum_tol: float,
):
    """Validate and stack per-modality matrices into (M,N,K), mask (M,N), Q (M,N)."""
    unknown = set(probs) - set(modalities)
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}; expected {list(modalities)}")
    n = None
    for name, arr in probs.items():
        if arr is not None:
            n = np.asarray(arr).shape[0] if np.ndim(arr) else None
            break
    if n is None:
        n = 0
    M = len(modalities)
    P = np.full((M, n, K), 1.0 / K)
    avail = np.zeros((M, n), dtype=bool)
    Q = np.ones((M, n))
    reasons = np.full((M, n), "missing", dtype=object)
    for i, name in enumerate(modalities):
        arr = probs.get(name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] != n:
            raise ValueError(f"modality {name!r} has {arr.shape[0]} rows, expected {n}")
        clean, valid, reason = validate_prob_rows(arr, K, sum_tol)
        P[i] = clean
        avail[i] = valid
        reasons[i] = reason
        q = None if quality is None else quality.get(name)
        if q is not None:
            qa = np.asarray(q, dtype=float)
            if qa.shape != (n,):
                raise ValueError(f"quality for {name!r} must have shape ({n},)")
            qa = np.where(np.isfinite(qa), np.clip(qa, 0.0, 1.0), 1.0)
            Q[i] = qa
    return P, avail, Q, reasons


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def equal_weight_fusion(
    inputs: Mapping[str, InputLike],
    labels: Sequence[str] = MELD_LABELS,
    modalities: Sequence[str] = MODALITIES,
    sum_tol: float = 1e-3,
) -> FusionResult:
    """Reference baseline: plain average of the available, valid modalities.

    Missing or invalid modalities are skipped (never counted as neutral), and
    the average is over the modalities that remain. This is the simple method
    the reliability-aware fuser is compared against.
    """
    unknown = set(inputs) - set(modalities)
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}; expected {list(modalities)}")
    K = len(labels)
    used, missing, warnings = [], [], []
    rows = []
    for name in modalities:
        mi = _coerce(inputs.get(name))
        if mi.probs is None:
            missing.append(name)
            continue
        clean, valid, reason = validate_prob_rows(np.asarray(mi.probs, dtype=float)[None, :], K, sum_tol)
        if not valid[0]:
            missing.append(name)
            warnings.append(f"{name}: dropped ({reason[0]})")
            continue
        if reason[0] != "ok":
            warnings.append(f"{name}: {reason[0]}")
        used.append(name)
        rows.append(clean[0])
    return _finish(
        np.mean(rows, axis=0) if rows else None,
        labels, used, missing, {n: 1.0 / len(used) for n in used},
        0.0, "equal_weight", warnings,
        conflict_score=_pairwise_mean_jsd(np.array(rows)) if len(rows) > 1 else 0.0,
        conflict_threshold=None, rows=rows,
    )


def _pairwise_mean_jsd(rows: np.ndarray) -> float:
    m = len(rows)
    if m < 2:
        return 0.0
    vals = [float(_jsd(rows[i], rows[j])) for i in range(m) for j in range(i + 1, m)]
    return float(np.mean(vals))


def _finish(
    probs, labels, used, missing, weights, min_confidence, method,
    warnings, conflict_score, conflict_threshold, rows,
) -> FusionResult:
    K = len(labels)
    if probs is None:
        return FusionResult(
            label=None, label_index=-1, probs=np.full(K, 1.0 / K), confidence=0.0,
            margin=0.0, available=[], missing=list(missing), weights={},
            conflict_score=0.0, conflict=False, uncertain=True, method=method,
            warnings=list(warnings) + ["no usable modality"],
        )
    idx = int(np.argmax(probs))
    top2 = np.sort(probs)[-2:]
    conflict = False
    if conflict_threshold is not None and len(rows) > 1:
        argmaxes = {int(np.argmax(r)) for r in rows}
        conflict = len(argmaxes) > 1 and conflict_score >= conflict_threshold
    confidence = float(probs[idx])
    return FusionResult(
        label=labels[idx], label_index=idx, probs=np.asarray(probs), confidence=confidence,
        margin=float(top2[1] - top2[0]), available=list(used), missing=list(missing),
        weights=dict(weights), conflict_score=float(conflict_score), conflict=bool(conflict),
        uncertain=bool(conflict or confidence < min_confidence), method=method,
        warnings=list(warnings),
    )


# --------------------------------------------------------------------------- #
# Reliability-aware fuser
# --------------------------------------------------------------------------- #
class ReliabilityAwareFuser:
    """Calibrated, reliability-weighted decision-level fusion.

    Parameters
    ----------
    labels : class names; every probability vector must follow this order.
    modalities : modality names (order is fixed for batch arrays).
    pooling : ``"linear"`` -- weighted mean of probabilities (the formula in the
        project brief). ``"log"`` -- weighted geometric pooling (product of
        experts); weights are scaled to average 1 so equal weights give the
        plain product of experts. Log pooling lets a confident, reliable
        modality overrule several unsure ones.
    confidence_power, quality_power, consistency_power : exponents ``gamma``,
        ``beta``, ``lambda`` above. 0 disables that factor.
    use_priors : multiply by the fitted per-modality prior (needs ``fit``).
    use_calibration : apply fitted temperatures (needs ``fit``).
    weight_floor : lower bound on every factor so a weight never hits exactly 0.
    conflict_threshold : mean pairwise JSD (bits) above which differing
        arg-maxes are reported as a conflict.
    min_confidence : fused confidence below this marks the result ``uncertain``.
    smoothing : weight of a uniform distribution mixed into each modality before
        log pooling, so a probability of exactly 0 is not an absolute veto.
    """

    def __init__(
        self,
        labels: Sequence[str] = MELD_LABELS,
        modalities: Sequence[str] = MODALITIES,
        pooling: str = "linear",
        confidence_power: float = 1.0,
        quality_power: float = 1.0,
        consistency_power: float = 0.0,
        use_priors: bool = True,
        use_calibration: bool = True,
        weight_floor: float = 1e-3,
        conflict_threshold: float = 0.25,
        min_confidence: float = 0.0,
        sum_tol: float = 1e-3,
        smoothing: float = 0.01,
    ):
        if pooling not in ("linear", "log"):
            raise ValueError("pooling must be 'linear' or 'log'")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)
        self.labels = tuple(labels)
        self.modalities = tuple(modalities)
        self.pooling = pooling
        self.confidence_power = float(confidence_power)
        self.quality_power = float(quality_power)
        self.consistency_power = float(consistency_power)
        self.use_priors = use_priors
        self.use_calibration = use_calibration
        self.weight_floor = float(weight_floor)
        self.conflict_threshold = float(conflict_threshold)
        self.min_confidence = float(min_confidence)
        self.sum_tol = float(sum_tol)
        self.temperatures: Dict[str, float] = {m: 1.0 for m in self.modalities}
        self.priors: Dict[str, float] = {m: 1.0 for m in self.modalities}
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, dev_probs: Mapping[str, np.ndarray], y_dev: np.ndarray) -> "ReliabilityAwareFuser":
        """Fit per-modality temperature and prior on a DEVELOPMENT split.

        ``dev_probs[name]`` is ``(N, K)``; rows that are missing/invalid are
        ignored. The prior is the modality's macro-F1 on its valid rows,
        rescaled so priors average to 1. Do not call this with test data.
        """
        y_dev = np.asarray(y_dev, dtype=int)
        K = len(self.labels)
        f1s = {}
        for name in self.modalities:
            arr = dev_probs.get(name)
            if arr is None:
                continue
            clean, valid, _ = validate_prob_rows(np.asarray(arr, dtype=float), K, self.sum_tol)
            if valid.sum() == 0:
                continue
            Pv, yv = clean[valid], y_dev[valid]
            self.temperatures[name] = fit_temperature(Pv, yv)
            f1s[name] = max(macro_f1(yv, Pv.argmax(axis=1), K), self.weight_floor)
        if f1s:
            mean = np.mean(list(f1s.values()))
            for name, f in f1s.items():
                self.priors[name] = f / mean
        self.fitted = True
        return self

    # ------------------------------------------------------------ batch core
    def fuse_batch(
        self,
        probs: Mapping[str, Optional[np.ndarray]],
        quality: Optional[Mapping[str, Optional[np.ndarray]]] = None,
    ) -> BatchResult:
        """Fuse ``N`` samples at once.

        ``probs[name]`` is ``(N, K)``; an all-NaN row (or a ``None`` entry)
        marks the modality as missing for that sample. ``quality[name]`` is an
        optional ``(N,)`` array in [0, 1].
        """
        K = len(self.labels)
        M = len(self.modalities)
        P, avail, Q, _ = _stack(probs, quality, self.modalities, K, self.sum_tol)
        N = P.shape[1]

        if self.use_calibration:
            for i, name in enumerate(self.modalities):
                P[i] = apply_temperature(P[i], self.temperatures[name])

        # --- reliability factors -------------------------------------------
        floor = self.weight_floor
        conf = np.clip(1.0 - _entropy_norm(P), floor, 1.0)                # (M, N)

        jsd = np.zeros((M, M, N))
        for i in range(M):
            for j in range(i + 1, M):
                jsd[i, j] = jsd[j, i] = _jsd(P[i], P[j])
        pair = avail[:, None, :] & avail[None, :, :] & ~np.eye(M, dtype=bool)[:, :, None]
        cnt = pair.sum(axis=1)                                             # (M, N)
        mean_jsd = (jsd * pair).sum(axis=1) / np.maximum(cnt, 1)
        cons = np.where(cnt > 0, np.clip(1.0 - mean_jsd, floor, 1.0), 1.0)

        prior = np.array([self.priors[m] if self.use_priors else 1.0 for m in self.modalities])
        w = (
            prior[:, None]
            * conf ** self.confidence_power
            * np.clip(Q, floor, 1.0) ** self.quality_power
            * cons ** self.consistency_power
        )
        w = np.where(avail, np.maximum(w, floor * prior[:, None]), 0.0)
        total = w.sum(axis=0)
        n_avail = avail.sum(axis=0)
        wn = np.divide(w, total, out=np.zeros_like(w), where=total > 0)

        # --- pooling ---------------------------------------------------------
        if self.pooling == "linear":
            fused = np.einsum("mn,mnk->nk", wn, P)
        else:
            wlog = wn * n_avail  # average weight 1 over available modalities
            # Mix with uniform first so one modality's ~0 probability for a class
            # (e.g. a text model that has no "neutral" class) cannot veto it.
            Ps = (1.0 - self.smoothing) * P + self.smoothing / K
            logp = np.einsum("mn,mnk->nk", wlog, np.log(np.clip(Ps, _EPS, 1.0)))
            fused = _softmax_rows(logp)
        none = n_avail == 0
        fused[none] = 1.0 / K

        # --- conflict --------------------------------------------------------
        n_pairs = np.maximum(n_avail * (n_avail - 1) / 2.0, 1)
        conflict_score = (jsd * pair).sum(axis=(0, 1)) / 2.0 / n_pairs
        argmaxes = P.argmax(axis=2)                                        # (M, N)
        a_masked = np.where(avail, argmaxes, -1)
        first = np.where(avail.any(axis=0), a_masked.max(axis=0), -1)
        differ = (avail & (argmaxes != first[None, :])).any(axis=0)
        conflict = differ & (n_avail > 1) & (conflict_score >= self.conflict_threshold)

        pred = np.where(none, -1, fused.argmax(axis=1))
        confidence = np.where(none, 0.0, fused.max(axis=1))
        uncertain = none | conflict | (confidence < self.min_confidence)
        return BatchResult(
            probs=fused, pred=pred, confidence=confidence, weights=wn, available=avail,
            conflict_score=conflict_score, conflict=conflict, uncertain=uncertain,
            modalities=self.modalities,
        )

    # ------------------------------------------------------------ single
    def fuse(self, inputs: Mapping[str, InputLike]) -> FusionResult:
        """Fuse one sample. Never raises on bad model outputs; drops them and warns."""
        unknown = set(inputs) - set(self.modalities)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)}; expected {list(self.modalities)}")
        K = len(self.labels)
        probs: Dict[str, Optional[np.ndarray]] = {}
        quality: Dict[str, np.ndarray] = {}
        warnings: List[str] = []
        for name in self.modalities:
            mi = _coerce(inputs.get(name))
            if mi.probs is None:
                probs[name] = None
                continue
            arr = np.asarray(mi.probs, dtype=float)
            if arr.shape != (K,):
                raise ValueError(f"{name}: expected {K} class probabilities, got shape {arr.shape}")
            probs[name] = arr[None, :]
            q, qwarn = validate_quality(mi.quality)
            if qwarn:
                warnings.append(f"{name}: {qwarn}")
            if q is not None:
                quality[name] = np.array([q])

        # Re-run validation here only to report reasons; the batch path repeats it.
        for name in self.modalities:
            if probs[name] is None:
                continue
            _, valid, reason = validate_prob_rows(probs[name], K, self.sum_tol)
            if not valid[0]:
                warnings.append(f"{name}: dropped ({reason[0]})")
            elif reason[0] != "ok":
                warnings.append(f"{name}: {reason[0]}")

        # Batch path needs equal row counts; absent modalities stay None.
        if all(v is None for v in probs.values()):
            return _finish(None, self.labels, [], list(self.modalities), {},
                           self.min_confidence, f"reliability_{self.pooling}", warnings,
                           0.0, self.conflict_threshold, [])
        b = self.fuse_batch(probs, quality)
        used = [m for i, m in enumerate(self.modalities) if b.available[i, 0]]
        missing = [m for m in self.modalities if m not in used]
        if not used:
            return _finish(None, self.labels, [], missing, {}, self.min_confidence,
                           f"reliability_{self.pooling}", warnings, 0.0,
                           self.conflict_threshold, [])
        weights = {m: float(b.weights[i, 0]) for i, m in enumerate(self.modalities) if m in used}
        idx = int(b.pred[0])
        top2 = np.sort(b.probs[0])[-2:]
        return FusionResult(
            label=self.labels[idx], label_index=idx, probs=b.probs[0],
            confidence=float(b.confidence[0]), margin=float(top2[1] - top2[0]),
            available=used, missing=missing, weights=weights,
            conflict_score=float(b.conflict_score[0]), conflict=bool(b.conflict[0]),
            uncertain=bool(b.uncertain[0]), method=f"reliability_{self.pooling}",
            warnings=warnings,
        )
