"""All tunable parameters of the fusion module in one documented place.

Every field carries a ``basis`` saying *why* its default is what it is:

* ``evidence``   -- chosen because of measurements (synthetic ones unless stated)
* ``assumption`` -- a modelling choice with no supporting measurement yet
* ``engineering``-- numerical safeguard / bookkeeping, not a modelling choice

``FusionConfig.docs()`` returns this table; ``fusion/README.md`` embeds it.
Nothing here is tuned on real data: no real model predictions exist yet.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Tuple

from .errors import FusionConfigError
from .labels import MELD_LABELS, MODALITIES


def _f(default, basis: str, doc: str):
    return field(default=default, metadata={"basis": basis, "doc": doc})


@dataclass(frozen=True)
class FusionConfig:
    labels: Tuple[str, ...] = _f(
        MELD_LABELS, "engineering",
        "Class names; the order of every probability vector. Must match the models' output order.")
    modalities: Tuple[str, ...] = _f(
        MODALITIES, "engineering", "Modality names accepted as input keys.")
    modality_support: Mapping[str, Tuple[str, ...]] = field(
        default_factory=dict,
        metadata={
            "basis": "engineering",
            "doc": ("Per-modality list of the classes the model can actually predict. Modalities not "
                    "listed support all classes. Columns outside the support are ignored, never read "
                    "as 0. See fusion/DESIGN_partial_support.md.")})
    pooling: str = _f(
        "linear", "assumption",
        "'linear' = weighted mean of probabilities (project-brief formula). 'log' = weighted product "
        "of experts. Log scored higher on SIMULATED data (+0.023 mean macro-F1, better in 8 of 9 "
        "scenarios, never worse), but that simulation assumes independent errors, which favours log "
        "by construction; choose after correlated-error and real-dev results.")
    use_calibration: bool = _f(
        True, "evidence",
        "Temperature-scale each modality (fit on dev data only; needs fit(), an un-fitted fuser warns "
        "once and applies nothing). Switching it off cost 0.038 macro-F1 on average in the "
        "simulation (worse in 8 of 9 scenarios) -- but the simulated models are over-confident by "
        "construction, so real-data benefit is unverified.")
    use_priors: bool = _f(
        False, "evidence",
        "Multiply weights by each modality's dev macro-F1. Simulated: -0.004 mean macro-F1 with "
        "linear pooling (worse in 5 of 9 scenarios, best case +0.003) and -0.002 with log. F1 values "
        "from different class subsets are also not directly comparable. Revisit in the "
        "prior-correction stage.")
    confidence_power: float = _f(
        0.0, "evidence",
        "Exponent on (1 - normalised entropy) of the calibrated prediction. 0 = off. Simulated: "
        "-0.016 (linear) and -0.012 (log) mean macro-F1, worse in 8 of 9 scenarios; entropy is not a "
        "validated measure of correctness.")
    quality_power: float = _f(
        0.0, "assumption",
        "Exponent on an externally supplied quality score in [0,1] (audio SNR, face-detected ratio). "
        "0 = off. Simulated: it helps only where the supplied score is informative (+0.015 linear / "
        "+0.031 log in the noisy-audio scenario with a known quality, 5/5 draws) and hurts otherwise "
        "(down to -0.013); mean -0.003 (linear), +0.001 (log). No real quality signal exists yet; opt "
        "in once one does and has been validated on dev data.")
    consistency_power: float = _f(
        0.0, "evidence",
        "Exponent on agreement with the other modalities. 0 = off. Simulated: helps only under "
        "injected conflicts (+0.014 linear / +0.008 log), flat or slightly negative elsewhere (mean "
        "-0.002 / -0.003); cannot work with 2 modalities (weights come out equal).")
    weight_floor: float = _f(
        1e-3, "engineering", "Lower bound on each weight factor so no weight is exactly 0.")
    smoothing: float = _f(
        0.01, "assumption",
        "Weight of a uniform distribution mixed into each modality before log pooling, so a tiny "
        "probability is not an absolute veto. Untuned.")
    conflict_threshold: float = _f(
        0.25, "assumption",
        "Mean pairwise Jensen-Shannon divergence (bits) above which differing arg-maxes are flagged "
        "as a conflict. NOT justified: the flag was not discriminative in simulation. Slated for "
        "replacement by a dev-derived uncertainty/abstention rule (P1).")
    min_confidence: float = _f(
        0.0, "engineering", "Fused confidence below this marks the result 'uncertain'. 0 = never.")
    sum_tol: float = _f(
        1e-3, "engineering",
        "Rows whose sum differs from 1 by more than this are flagged 'renormalized'.")
    min_fit_samples: int = _f(
        30, "engineering",
        "Minimum number of usable dev rows needed to fit a modality's temperature and prior; with "
        "fewer, both stay at 1.0 and fit() warns. A floor against fitting noise, not a recommendation "
        "(a few hundred rows is the realistic minimum for a stable temperature).")
    strict: bool = _f(
        False, "engineering",
        "True: raise FusionInputError for invalid model outputs (NaN/negative/zero-sum rows, mass on "
        "a declared-unsupported class, non-finite or out-of-range quality) instead of dropping them "
        "with a status code. Missing modalities (None / all-NaN rows) are always allowed.")

    # ------------------------------------------------------------------ checks
    def __post_init__(self):
        set_ = lambda k, v: object.__setattr__(self, k, v)  # frozen dataclass
        set_("labels", tuple(str(x) for x in self.labels))
        set_("modalities", tuple(str(x) for x in self.modalities))
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise FusionConfigError("labels must be >= 2 unique names")
        if len(self.modalities) < 1 or len(set(self.modalities)) != len(self.modalities):
            raise FusionConfigError("modalities must be unique names")
        if self.pooling not in ("linear", "log"):
            raise FusionConfigError("pooling must be 'linear' or 'log'")
        def real(name, v):       # numpy scalars are fine; bool is not a number here
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
                raise FusionConfigError(f"{name} must be a finite number, got {v!r}")
            return float(v)

        for name in ("confidence_power", "quality_power", "consistency_power"):
            if real(name, getattr(self, name)) < 0:
                raise FusionConfigError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if not 0.0 <= real("smoothing", self.smoothing) < 1.0:
            raise FusionConfigError("smoothing must be in [0, 1)")
        if not 0.0 < real("weight_floor", self.weight_floor) < 1.0:
            raise FusionConfigError("weight_floor must be in (0, 1)")
        if not 0.0 < real("sum_tol", self.sum_tol) <= 0.1:
            raise FusionConfigError("sum_tol must be in (0, 0.1]; a larger tolerance would hide "
                                    "probability mass on unsupported classes and badly scaled rows")
        if not 0.0 <= real("min_confidence", self.min_confidence) <= 1.0:
            raise FusionConfigError("min_confidence must be in [0, 1]")
        if not 0.0 <= real("conflict_threshold", self.conflict_threshold) <= 1.0:
            raise FusionConfigError("conflict_threshold must be in [0, 1] (Jensen-Shannon divergence in bits)")
        if (isinstance(self.min_fit_samples, bool) or not isinstance(self.min_fit_samples, numbers.Integral)
                or self.min_fit_samples < 1):
            raise FusionConfigError(f"min_fit_samples must be an integer >= 1, got {self.min_fit_samples!r}")
        sup: Dict[str, Tuple[str, ...]] = {}
        for mod, labs in dict(self.modality_support).items():
            if mod not in self.modalities:
                raise FusionConfigError(f"modality_support names unknown modality {mod!r}")
            labs = tuple(str(x) for x in labs)
            bad = [x for x in labs if x not in self.labels]
            if bad:
                raise FusionConfigError(f"modality_support[{mod!r}] has labels not in labels: {bad}")
            if len(set(labs)) != len(labs) or len(labs) < 2:
                raise FusionConfigError(f"modality_support[{mod!r}] needs >= 2 unique labels")
            sup[mod] = labs
        partial = {m: frozenset(l) for m, l in sup.items() if len(l) < len(self.labels)}
        if len(set(partial.values())) > 1:
            raise FusionConfigError(
                "modalities with partial class support must all share the same supported set "
                f"(got {sorted(sorted(s) for s in set(partial.values()))}); mixed partial supports "
                "are not implemented")
        set_("modality_support", sup)

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "modality_support":
                v = {k: list(x) for k, x in v.items()}
            elif isinstance(v, tuple):
                v = list(v)
            d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FusionConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise FusionConfigError(f"unknown config options: {sorted(unknown)}")
        return cls(**dict(d))

    @classmethod
    def docs(cls) -> List[Dict[str, Any]]:
        """Rows ``{name, default, basis, doc}`` describing every option."""
        rows = []
        for f in fields(cls):
            default = f.default if f.default is not field else None
            if f.default_factory is not type(f.default_factory) and f.name == "modality_support":
                default = {}
            rows.append({"name": f.name, "default": default,
                         "basis": f.metadata.get("basis", ""), "doc": f.metadata.get("doc", "")})
        return rows

    @classmethod
    def equal_weight_baseline(cls, **overrides) -> "FusionConfig":
        """Plain average of the available modalities: no calibration, no weighting."""
        from dataclasses import replace
        return replace(cls(), use_calibration=False, use_priors=False, confidence_power=0.0,
                       quality_power=0.0, consistency_power=0.0, pooling="linear", **overrides)
