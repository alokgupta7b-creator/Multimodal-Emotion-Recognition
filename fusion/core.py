"""Decision-level fusion of text / audio / face emotion probabilities.

Public API
----------
``FusionConfig``                          every tunable option, with documented defaults
``ReliabilityAwareFuser(config)``         calibrated, reliability-weighted fusion
    ``.fit(dev_probs, y_dev)``            fit temperatures (+ modality priors) on a DEV split
    ``.fuse(inputs)``                     one sample  -> ``FusionResult``
    ``.fuse_batch(probs, quality)``       many samples -> ``BatchResult``
``equal_weight_fusion(inputs)``           reference baseline (plain average)

``fuse`` is a thin wrapper around ``fuse_batch``: there is ONE validation and
fusion code path, so single-sample and batch results cannot disagree.

Input for one sample is a mapping ``{"text": ..., "audio": ..., "face": ...}``
where each value is a probability vector over ``labels`` (list/array), a
``ModalityInput`` (adds an optional ``quality`` in [0, 1]), or ``None`` when the
modality is unavailable. Omitted modalities count as unavailable.

Weight of modality ``m`` on one sample (every factor is optional; the default
configuration uses none of them, i.e. calibrated equal weights)::

    w_m = prior_m * conf_m**gamma * quality_m**beta * consistency_m**lambda

See ``fusion/config.py`` for the meaning and evidence behind each default and
``fusion/DESIGN_partial_support.md`` for modalities that cannot predict every
class (e.g. the text model, which has no ``neutral``/``disgust``).
"""
from __future__ import annotations

import warnings as _warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .calibration import (
    T_MAX, T_MIN, apply_temperature, apply_temperature_masked, fit_temperature, temperature_at_bound,
)
from .config import FusionConfig
from .errors import FusionConfigError, FusionInputError
from .labels import MELD_LABELS, MODALITIES
from .metrics import macro_f1
from .validation import (
    INVALID, MISSING, Q_CLIPPED, Q_INVALID, RENORMALIZED, USABLE,
    validate_prob_rows, validate_quality_array,
)

_EPS = 1e-12
# Two modalities whose probability mass on the classes both can predict is below this have no
# measurable disagreement: the conditional distribution would be a ratio of numerical noise.
_MIN_SHARED_MASS = 1e-6


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
    probs: np.ndarray               # fused distribution; NaN = class not scored (see partial_only),
                                    # or every entry NaN when there is no prediction (none invented)
    confidence: float               # max fused probability (0.0 if no prediction)
    margin: float                   # top-1 minus top-2 fused probability
    available: List[str]            # modalities that contributed
    missing: List[str]              # modalities absent or dropped as invalid
    weights: Dict[str, float]       # normalised weights over the available modalities
    conflict_score: float           # mean pairwise JSD (bits, in [0, 1]) among available
    conflict: bool                  # modalities disagree (see FusionConfig.conflict_threshold)
    uncertain: bool                 # conflict, no prediction, or confidence < min_confidence
    method: str
    warnings: List[str] = field(default_factory=list)
    status: Dict[str, str] = field(default_factory=dict)          # per-modality validation status
    quality_status: Dict[str, str] = field(default_factory=dict)  # per-modality quality status
    partial_only: bool = False      # only partial-class-support modalities were available

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["probs"] = [None if not np.isfinite(x) else float(x) for x in self.probs]
        return d


@dataclass
class BatchResult:
    probs: np.ndarray               # (N, K); NaN = class not scored (partial_only rows) or no prediction
    pred: np.ndarray                # (N,) int, -1 = no prediction
    confidence: np.ndarray          # (N,)
    weights: np.ndarray             # (M, N) normalised over available modalities
    available: np.ndarray           # (M, N) bool
    conflict_score: np.ndarray      # (N,)
    conflict: np.ndarray            # (N,) bool
    uncertain: np.ndarray           # (N,) bool
    modalities: tuple
    status: np.ndarray = None           # (M, N) object: validation status per modality/sample
    quality_status: np.ndarray = None   # (M, N) object: ok / unknown / invalid / clipped
    partial_only: np.ndarray = None     # (N,) bool
    labels: tuple = ()

    def summary(self) -> dict:
        """Counts a reader can check: why modalities were dropped, how many rows had no prediction."""
        per_mod = {m: dict(Counter(self.status[i].tolist())) for i, m in enumerate(self.modalities)}
        q = {m: {k: v for k, v in Counter(self.quality_status[i].tolist()).items() if k in (Q_CLIPPED, Q_INVALID)}
             for i, m in enumerate(self.modalities)}
        return {
            "n": int(len(self.pred)),
            "status": per_mod,
            "quality_issues": {m: c for m, c in q.items() if c},
            "no_prediction": int((self.pred == -1).sum()),
            "partial_only": int(self.partial_only.sum()),
            "conflict": int(self.conflict.sum()),
            "uncertain": int(self.uncertain.sum()),
        }


# --------------------------------------------------------------------------- #
# Numerical helpers
# --------------------------------------------------------------------------- #
def _entropy_norm(P: np.ndarray, k: int) -> np.ndarray:
    """Shannon entropy normalised by log(k), in [0, 1], along the last axis."""
    ent = -(P * np.log(np.clip(P, _EPS, 1.0))).sum(axis=-1)
    return np.clip(ent / np.log(k), 0.0, 1.0)


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


def _cond(P: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Distribution restricted to ``cols`` and renormalised; uniform if it has no mass there."""
    sub = P[..., cols]
    tot = sub.sum(axis=-1, keepdims=True)
    k = sub.shape[-1]
    return np.where(tot > _EPS, sub / np.clip(tot, _EPS, None), 1.0 / k)


def _pool(Pm: np.ndarray, w: np.ndarray, pooling: str, smoothing: float) -> np.ndarray:
    """Pool distributions ``Pm`` (M, n, k) with weights ``w`` (M, n); weight 0 = not participating.

    linear: weighted mean. log: weighted geometric mean (product of experts) with weights
    scaled to average 1 over participants, after mixing a little uniform mass into every
    expert so that a tiny probability is not an absolute veto.
    """
    k = Pm.shape[-1]
    total = w.sum(axis=0)
    wn = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    if pooling == "linear":
        out = np.einsum("mn,mnk->nk", wn, Pm)
    else:
        n_used = (w > 0).sum(axis=0)
        Ps = (1.0 - smoothing) * Pm + smoothing / k
        logp = np.einsum("mn,mnk->nk", wn * n_used, np.log(np.clip(Ps, _EPS, 1.0)))
        out = _softmax_rows(logp)
    out[total <= 0] = 1.0 / k
    return out


def _as_label_indices(y, labels: Sequence[str]) -> np.ndarray:
    """Integer class indices from ints or label names; raises FusionInputError otherwise."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise FusionInputError(f"labels must be 1-D, got shape {arr.shape}")
    K = len(labels)
    if arr.dtype.kind in "US" or arr.dtype == object:
        lookup = {l: i for i, l in enumerate(labels)}
        bad = sorted({str(v) for v in arr if str(v) not in lookup})
        if bad:
            raise FusionInputError(f"labels not in the label set {list(labels)}: {bad[:5]}")
        return np.array([lookup[str(v)] for v in arr], dtype=int)
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all() or not np.equal(np.mod(arr, 1), 0).all():
            raise FusionInputError("labels must be integers or label names (got non-integer / NaN values)")
    elif arr.dtype.kind not in "iub":
        raise FusionInputError(f"labels must be integers or label names, got dtype {arr.dtype}")
    arr = arr.astype(int)
    if arr.size and (arr.min() < 0 or arr.max() >= K):
        raise FusionInputError(
            f"integer labels must be in [0, {K - 1}]; found min {arr.min()} and max {arr.max()}")
    return arr


# --------------------------------------------------------------------------- #
# Reliability-aware fuser
# --------------------------------------------------------------------------- #
class ReliabilityAwareFuser:
    """Calibrated, optionally reliability-weighted decision-level fusion.

    ``ReliabilityAwareFuser(FusionConfig(...))`` or, equivalently, keyword
    overrides ``ReliabilityAwareFuser(pooling="log", consistency_power=1.0)``.
    All options and the evidence behind each default: ``FusionConfig.docs()``.
    """

    def __init__(self, config: Optional[FusionConfig] = None, **overrides):
        if config is None:
            config = FusionConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config
        K = len(config.labels)
        self.temperatures: Dict[str, float] = {m: 1.0 for m in config.modalities}
        self.priors: Dict[str, float] = {m: 1.0 for m in config.modalities}
        self.fit_report: Dict[str, dict] = {}
        self.fitted = False
        self._warned_unfitted = False
        # Boolean support mask per modality (True = the model can predict that class).
        self._support: Dict[str, np.ndarray] = {}
        for m in config.modalities:
            mask = np.ones(K, dtype=bool)
            if m in config.modality_support:
                mask = np.array([l in config.modality_support[m] for l in config.labels])
            self._support[m] = mask
        self._is_partial = np.array([not self._support[m].all() for m in config.modalities])
        partial_sets = [self._support[m] for m, p in zip(config.modalities, self._is_partial) if p]
        self._partial_mask: Optional[np.ndarray] = partial_sets[0] if partial_sets else None

    def __getattr__(self, name):  # config options stay readable as attributes (f.pooling, ...)
        cfg = self.__dict__.get("config")
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
        raise AttributeError(name)

    @property
    def method(self) -> str:
        c = self.config
        plain = (not c.use_calibration and not c.use_priors and c.pooling == "linear"
                 and c.confidence_power == 0 and c.quality_power == 0 and c.consistency_power == 0)
        return "equal_weight" if plain else f"reliability_{c.pooling}"

    # ------------------------------------------------------------------ fit
    def fit(self, dev_probs: Mapping[str, np.ndarray], y_dev) -> "ReliabilityAwareFuser":
        """Fit per-modality temperature and prior on a DEVELOPMENT split.

        ``dev_probs[name]`` is ``(N, K)`` aligned with ``y_dev`` (integer indices or label
        names). Rows that are missing/invalid are ignored. For a partial-support modality
        only samples whose true class it can predict are used, and temperature/F1 are
        computed on its conditional distribution over its supported classes (so they are
        measured on a subset that excludes the classes it cannot see).
        Never call this with test data. Raises ``FusionInputError`` on misaligned or
        invalid inputs; the outcome per modality is stored in ``self.fit_report``.
        """
        cfg = self.config
        K = len(cfg.labels)
        unknown = set(dev_probs) - set(cfg.modalities)
        if unknown:
            raise FusionInputError(f"unknown modalities {sorted(unknown)}; expected {list(cfg.modalities)}")
        y = _as_label_indices(y_dev, cfg.labels)
        report, f1s = {}, {}
        # Built from scratch and assigned only at the end: a re-fit never keeps state from an
        # earlier fit, and a fit that raises leaves the fuser exactly as it was.
        temps = {m: 1.0 for m in cfg.modalities}
        priors = {m: 1.0 for m in cfg.modalities}
        for name in cfg.modalities:
            arr = dev_probs.get(name)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != len(y):
                raise FusionInputError(
                    f"{name}: dev probabilities have shape {arr.shape} but y_dev has {len(y)} labels; "
                    f"they must be row-aligned with shape (N, {K})")
            mask = self._support[name]
            clean, valid, _ = validate_prob_rows(arr, K, cfg.sum_tol, columns=mask)
            in_support = mask[y]
            use = valid & in_support
            info = {"n_rows": int(len(y)), "n_valid": int(valid.sum()), "n_used": int(use.sum()),
                    "supported_classes": [l for l, s in zip(cfg.labels, mask) if s]}
            if use.sum() < cfg.min_fit_samples:
                n_use = int(use.sum())
                info["note"] = (
                    "no usable dev rows; temperature and prior stay at 1.0" if n_use == 0 else
                    f"only {n_use} usable dev rows (< min_fit_samples={cfg.min_fit_samples}); "
                    "temperature and prior stay at 1.0")
                _warnings.warn(f"{name}: {info['note']}", UserWarning, stacklevel=2)
                report[name] = info
                continue
            cols = np.flatnonzero(mask)
            pos = {c: i for i, c in enumerate(cols)}
            Pv = clean[use][:, cols]
            yv = np.array([pos[c] for c in y[use]])
            T = fit_temperature(Pv, yv) if cfg.use_calibration else 1.0
            temps[name] = T
            at_bound = bool(cfg.use_calibration and temperature_at_bound(T))
            if at_bound:
                _warnings.warn(
                    f"{name}: fitted temperature {T:.3g} is at the search bound [{T_MIN}, {T_MAX}]; the dev "
                    "data is (nearly) separable or degenerate for this model, so this temperature is not "
                    "trustworthy. Check the dev predictions before relying on calibration.",
                    UserWarning, stacklevel=2)
            f1 = max(macro_f1(yv, Pv.argmax(axis=1), len(cols)), cfg.weight_floor)
            f1s[name] = f1
            info.update(temperature=float(T), temperature_at_bound=at_bound, macro_f1=float(f1))
            report[name] = info
        if f1s:
            mean = float(np.mean(list(f1s.values())))
            for name, f in f1s.items():
                priors[name] = f / mean
        self.temperatures, self.priors = temps, priors
        self.fit_report = report
        self.fitted = True
        return self

    # ------------------------------------------------------------ batch core
    def fuse_batch(
        self,
        probs: Mapping[str, Optional[np.ndarray]],
        quality: Optional[Mapping[str, Optional[np.ndarray]]] = None,
        n: Optional[int] = None,
    ) -> BatchResult:
        """Fuse ``N`` samples at once.

        ``probs[name]`` is ``(N, K)`` or ``None``; an all-NaN row marks the modality as
        missing for that sample. ``quality[name]`` is an optional ``(N,)`` array in [0, 1].
        ``n`` is only needed when every modality is ``None``. Raises ``FusionInputError``
        for unknown modality names, misaligned rows or wrong class counts. Invalid model
        outputs are dropped and reported in ``status`` (or raise if ``config.strict``).
        """
        cfg = self.config
        mods, K, M = cfg.modalities, len(cfg.labels), len(cfg.modalities)
        if (cfg.use_calibration or cfg.use_priors) and not self.fitted and not self._warned_unfitted:
            self._warned_unfitted = True
            _warnings.warn(
                "use_calibration/use_priors is on but fit() has not been called, so every temperature "
                "and prior is 1.0 (nothing is calibrated). Call fit() on a development split, or set "
                "use_calibration=False and use_priors=False to silence this.", UserWarning, stacklevel=2)
        quality = quality or {}
        for source, mapping in (("probs", probs), ("quality", quality)):
            unknown = set(mapping) - set(mods)
            if unknown:
                raise FusionInputError(
                    f"unknown modalities {sorted(unknown)} in {source}; expected {list(mods)}")

        rows = {name: np.asarray(a).shape[0] if np.ndim(a) else None
                for name, a in probs.items() if a is not None}
        if any(r is None for r in rows.values()):
            raise FusionInputError("probabilities must be 2-D (N, K); got a scalar")
        if len(set(rows.values())) > 1:
            raise FusionInputError(f"modalities have different row counts: {rows}")
        N = next(iter(rows.values())) if rows else (0 if n is None else int(n))

        P = np.full((M, N, K), 1.0 / K)
        avail = np.zeros((M, N), dtype=bool)
        status = np.full((M, N), MISSING, dtype=object)
        Q = np.ones((M, N))
        qstat = np.full((M, N), "unknown", dtype=object)
        for i, name in enumerate(mods):
            arr = probs.get(name)
            if arr is not None:
                clean, valid, reason = validate_prob_rows(
                    self._named(name, arr), K, cfg.sum_tol, columns=self._support[name])
                P[i], avail[i], status[i] = clean, valid, reason
            Q[i], qstat[i] = validate_quality_array(quality.get(name), N, name)

        if cfg.strict:
            self._raise_if_invalid(mods, status, qstat, avail)

        # ---- calibration (temperatures were fitted on dev data) ----------------------------
        if cfg.use_calibration:
            for i, name in enumerate(mods):
                T = self.temperatures[name]
                if self._is_partial[i]:
                    P[i] = apply_temperature_masked(P[i], T, self._support[name])
                else:
                    P[i] = apply_temperature(P[i], T)

        # ---- pairwise agreement, compared on the classes both models can predict ------------
        floor = cfg.weight_floor
        jsd = np.zeros((M, M, N))
        differ = np.zeros((M, M, N), dtype=bool)
        measurable = np.ones((M, M, N), dtype=bool)
        for i in range(M):
            for j in range(i + 1, M):
                common = self._support[mods[i]] & self._support[mods[j]]
                if common.all():
                    a, b = P[i], P[j]
                else:
                    a, b = _cond(P[i], common), _cond(P[j], common)
                    # A modality with ~no mass on the shared classes says nothing about them; do not
                    # let the uniform fallback of _cond() manufacture a disagreement score.
                    shared = ((P[i][:, common].sum(axis=1) > _MIN_SHARED_MASS)
                              & (P[j][:, common].sum(axis=1) > _MIN_SHARED_MASS))
                    measurable[i, j] = measurable[j, i] = shared
                jsd[i, j] = jsd[j, i] = _jsd(a, b)
                differ[i, j] = differ[j, i] = a.argmax(axis=1) != b.argmax(axis=1)
        pair = (avail[:, None, :] & avail[None, :, :] & ~np.eye(M, dtype=bool)[:, :, None]
                & measurable)
        # A full-support modality's agreement score must not depend on a partial-support one: the
        # partial modality (text) may only redistribute mass inside its own classes, never change the
        # base weights (and through them P(supported classes), P(neutral), P(disgust)).
        excl = (~self._is_partial)[:, None] & self._is_partial[None, :]       # row full, column partial
        pair_w = pair & ~excl[:, :, None]
        cnt = pair_w.sum(axis=1)
        mean_jsd = (jsd * pair_w).sum(axis=1) / np.maximum(cnt, 1)
        cons = np.where(cnt > 0, np.clip(1.0 - mean_jsd, floor, 1.0), 1.0)

        # ---- reliability weights ------------------------------------------------------------
        conf = np.empty((M, N))
        for i, name in enumerate(mods):
            mask = self._support[name]
            conf[i] = 1.0 - _entropy_norm(P[i][:, mask], int(mask.sum()))
        conf = np.clip(conf, floor, 1.0)
        prior = np.array([self.priors[m] if cfg.use_priors else 1.0 for m in mods])
        w = (prior[:, None]
             * conf ** cfg.confidence_power
             * np.clip(Q, floor, 1.0) ** cfg.quality_power
             * cons ** cfg.consistency_power)
        w = np.where(avail, np.maximum(w, floor * prior[:, None]), 0.0)
        total = w.sum(axis=0)
        wn = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        n_avail = avail.sum(axis=0)

        # ---- pooling --------------------------------------------------------------------------
        # Base fusion: the full-support modalities only, exactly as if no partial-support modality
        # existed (so with no such modality configured this is the whole pooling step).
        w_full = np.where(self._is_partial[:, None], 0.0, w)
        fused = _pool(P, w_full, cfg.pooling, cfg.smoothing)
        partial_only = np.zeros(N, dtype=bool)
        if self._partial_mask is not None:
            s = self._partial_mask
            part_av = (avail & self._is_partial[:, None]).any(axis=0)
            full_av = (avail & ~self._is_partial[:, None]).any(axis=0)
            both, only_partial = part_av & full_av, part_av & ~full_av
            if both.any() or only_partial.any():
                # A partial-support model (e.g. text, no neutral/disgust) outputs P(c | x, c in S).
                # It can say nothing about S-versus-rest, so it never moves mass between them:
                #   c outside S :  P(c) = P_base(c)                       (untouched)
                #   c in S      :  P(c) = P_base(S) * q(c)
                # where q updates the base's own within-S conditional with the text conditional(s).
                Pc = _cond(P, s)                                                  # (M, N, |S|)
                w_part = np.where(self._is_partial[:, None], w, 0.0)
                bS = _cond(fused, s)                                              # base | S
                mass_S = fused[:, s].sum(axis=1, keepdims=True)
                Wb = w_full.sum(axis=0)
                if cfg.pooling == "linear":
                    num = Wb[:, None] * bS + np.einsum("mn,mnk->nk", w_part, Pc)
                    q = num / np.clip(Wb + w_part.sum(axis=0), _EPS, None)[:, None]
                else:
                    # Bayes-style update: q ~ base|S * prod_i t_i^nu_i, nu_i = weight relative to
                    # the mean full-support weight (1 for equal weights). A uniform t is the identity.
                    n_full = np.maximum((w_full > 0).sum(axis=0), 1)
                    nu = w_part / np.clip(Wb / n_full, _EPS, None)                # (M, N)
                    Ps = (1.0 - cfg.smoothing) * Pc + cfg.smoothing / Pc.shape[-1]
                    logq = (np.log(np.clip(bS, _EPS, 1.0))
                            + np.einsum("mn,mnk->nk", nu, np.log(np.clip(Ps, _EPS, 1.0))))
                    q = _softmax_rows(logq)
                combo = fused.copy()
                combo[:, s] = mass_S * q
                fused[both] = combo[both]
                only = np.full_like(fused, np.nan)                 # no full-support evidence at all:
                only[:, s] = _pool(Pc, w_part, cfg.pooling, cfg.smoothing)   # restricted to S, flagged
                fused[only_partial] = only[only_partial]
                partial_only = only_partial

        none = n_avail == 0
        fused[none] = np.nan            # no modality, no prediction: do not invent a distribution
        partial_only = partial_only & ~none

        # ---- conflict -----------------------------------------------------------------------
        n_pairs = np.maximum(pair.sum(axis=(0, 1)) / 2.0, 1)
        conflict_score = (jsd * pair).sum(axis=(0, 1)) / 2.0 / n_pairs
        any_differ = ((differ & pair).sum(axis=(0, 1)) > 0)
        conflict = any_differ & (n_avail > 1) & (conflict_score >= cfg.conflict_threshold)

        scored = np.where(np.isnan(fused), -np.inf, fused)
        pred = np.where(none, -1, scored.argmax(axis=1))
        confidence = np.where(none, 0.0, scored.max(axis=1))
        uncertain = none | conflict | (confidence < cfg.min_confidence)
        return BatchResult(
            probs=fused, pred=pred, confidence=confidence, weights=wn, available=avail,
            conflict_score=conflict_score, conflict=conflict, uncertain=uncertain,
            modalities=mods, status=status, quality_status=qstat, partial_only=partial_only,
            labels=cfg.labels,
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _named(name: str, arr) -> np.ndarray:
        try:
            return np.asarray(arr, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FusionInputError(f"{name}: probabilities are not numeric ({exc})") from None

    @staticmethod
    def _raise_if_invalid(mods, status, qstat, avail):
        problems = []
        for i, name in enumerate(mods):
            c = Counter(s for s in status[i].tolist() if s in INVALID)
            if c:
                problems.append(f"{name}: {dict(c)}")
            qc = Counter(s for s, a in zip(qstat[i].tolist(), status[i].tolist())
                         if s in (Q_CLIPPED, Q_INVALID) and a not in (MISSING,))
            if qc:
                problems.append(f"{name} quality: {dict(qc)}")
        if problems:
            raise FusionInputError("strict mode: invalid model outputs -- " + "; ".join(problems))

    # ------------------------------------------------------------ single sample
    def fuse(self, inputs: Mapping[str, InputLike]) -> FusionResult:
        """Fuse one sample (wrapper around ``fuse_batch``; identical validation and maths).

        Bad model outputs are dropped with a warning (or raise if ``config.strict``).
        A wrong vector length or an unknown modality name raises ``FusionInputError``.
        """
        cfg = self.config
        K = len(cfg.labels)
        unknown = set(inputs) - set(cfg.modalities)
        if unknown:
            raise FusionInputError(f"unknown modalities {sorted(unknown)}; expected {list(cfg.modalities)}")
        probs: Dict[str, Optional[np.ndarray]] = {}
        quality: Dict[str, np.ndarray] = {}
        for name in cfg.modalities:
            raw = inputs.get(name)
            mi = raw if isinstance(raw, ModalityInput) else ModalityInput(probs=raw)
            if mi.probs is None:
                probs[name] = None
            else:
                arr = self._named(name, mi.probs)
                if arr.shape != (K,):
                    raise FusionInputError(f"{name}: expected {K} class probabilities, got shape {arr.shape}")
                probs[name] = arr[None, :]
            if mi.quality is not None:
                # A number (or a one-element array) is fine; NaN/inf is reported as 'invalid' by the
                # shared validation. Anything non-numeric is a caller bug and must not be hidden.
                try:
                    qa = np.asarray(mi.quality, dtype=float)
                except (TypeError, ValueError):
                    raise FusionInputError(f"{name}: quality is not numeric: {mi.quality!r}") from None
                if qa.size != 1:
                    raise FusionInputError(
                        f"{name}: quality must be a single number for one sample, got shape {qa.shape}")
                quality[name] = qa.reshape(1)
        b = self.fuse_batch(probs, quality, n=1)

        used = [m for i, m in enumerate(cfg.modalities) if b.available[i, 0]]
        missing = [m for m in cfg.modalities if m not in used]
        st = {m: str(b.status[i, 0]) for i, m in enumerate(cfg.modalities)}
        qs = {m: str(b.quality_status[i, 0]) for i, m in enumerate(cfg.modalities)}
        warnings: List[str] = []
        for m in cfg.modalities:
            if st[m] in INVALID:
                warnings.append(f"{m}: dropped ({st[m]})")
            elif st[m] == RENORMALIZED:
                warnings.append(f"{m}: renormalized")
            if st[m] != MISSING:
                if qs[m] == Q_CLIPPED:
                    warnings.append(f"{m}: quality outside [0, 1]; clipped")
                elif qs[m] == Q_INVALID:
                    warnings.append(f"{m}: quality is not finite; ignored")
        common = dict(method=self.method, status=st, quality_status=qs)
        if not used:
            return FusionResult(
                label=None, label_index=-1, probs=np.full(K, np.nan), confidence=0.0, margin=0.0,
                available=[], missing=missing, weights={}, conflict_score=0.0, conflict=False,
                uncertain=True, warnings=warnings + ["no usable modality"], **common)
        p = b.probs[0]
        scored = np.where(np.isnan(p), -np.inf, p)
        top2 = np.sort(scored)[-2:]
        idx = int(b.pred[0])
        if bool(b.partial_only[0]):
            covered = [l for l, s in zip(cfg.labels, self._partial_mask) if s]
            warnings.append(
                f"only partial-class-support modalities available ({', '.join(used)}): the prediction is "
                f"restricted to {covered}; the other classes were not scored and confidence is "
                f"conditional on those classes")
        return FusionResult(
            label=cfg.labels[idx], label_index=idx, probs=p, confidence=float(b.confidence[0]),
            margin=float(top2[1] - top2[0]) if np.isfinite(top2[0]) else float(top2[1]),
            available=used, missing=missing,
            weights={m: float(b.weights[i, 0]) for i, m in enumerate(cfg.modalities) if m in used},
            conflict_score=float(b.conflict_score[0]), conflict=bool(b.conflict[0]),
            uncertain=bool(b.uncertain[0]), warnings=warnings,
            partial_only=bool(b.partial_only[0]), **common)


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

    Same validation as the main fuser; missing or invalid modalities are skipped
    (never counted as neutral) and the average is over what remains. No calibration,
    no weighting. This is the method the reliability-aware fuser is compared against.
    """
    cfg = FusionConfig.equal_weight_baseline(labels=tuple(labels), modalities=tuple(modalities),
                                             sum_tol=sum_tol)
    return ReliabilityAwareFuser(cfg).fuse(inputs)
