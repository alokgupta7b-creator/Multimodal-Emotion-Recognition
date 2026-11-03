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
from dataclasses import MISSING, dataclass, field, fields
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
    model_priors: Mapping[str, Any] = field(
        default_factory=dict,
        metadata={
            "basis": "assumption",
            "doc": ("Per-modality class prior that the model's probabilities ASSUME (the prior of its "
                    "training distribution): 'uniform' (default) or one positive number per label, in label "
                    "order. class_weight='balanced' training -> uniform; unweighted training -> the training "
                    "class frequencies. It is divided out before pooling so a prior is not counted once per "
                    "model (Bayes: P(c|x_1..x_M) ~ pi_t(c) * prod_m p_m(c|x_m) / pi_m(c)). Declared by the "
                    "model owner, never estimated here; a wrong declaration makes results worse, not better. "
                    "Partial-support modalities must stay 'uniform'.")})
    target_prior: Any = _f(
        "uniform", "assumption",
        "Class prior of the FUSED output, applied exactly once: 'uniform' (default: no prior imposed; suits "
        "macro-F1 / balanced accuracy), 'dev' (class frequencies of the y_dev given to fit(), Laplace-smoothed; "
        "suits accuracy) or an explicit vector. This is a real accuracy / macro-F1 trade-off, not a free gain: "
        "on the real MELD text model alone (independent reproduction, text only) moving uniform -> dev "
        "frequencies raised test accuracy 0.516 -> 0.597 and lowered test macro-F1 0.378 -> 0.347.")
    target_prior_power: float = _f(
        1.0, "assumption",
        "Exponent alpha in [0, 1] applied to a non-uniform target prior (pi_t ~ pi^alpha): 0 = uniform, 1 = "
        "as given. Interpolates between the two objectives (text-only check on real MELD outputs: alpha=0.5 "
        "gave test accuracy 0.582 / macro-F1 0.379 vs 0.516 / 0.378 for uniform). Choose it on dev data "
        "only. Ignored when target_prior is 'uniform'. Setting any prior option also switches calibration to the prior-aware form: "
        "only the evidence log p - log pi_m is scaled by 1/T, and T is fitted against the dev class prior.")
    calibrate_fused: bool = _f(
        False, "evidence",
        "Fit one extra temperature on the FUSED dev output (after pooling), separately for each availability pattern "
        "(which modalities were used for a sample; dev data has all of them, so each pattern is fitted by pooling only that "
        "subset), and soften each fused output with the temperature of its own pattern. Argmax is unchanged; only confidence "
        "changes. Per-model calibration does not calibrate the pooled output when errors are correlated: simulated NLL 1.29 "
        "(log) vs 1.09 (linear) at rho=0.6; real MELD text+speech+face, test NLL/ECE 1.584/0.268 -> 1.343/0.042 (linear, "
        "uniform target) and 1.277/0.087 -> 1.214/0.072 (log, dev target). A single temperature fitted with all modalities "
        "present hurt missing-modality cases on real data (text alone NLL 1.366 -> 1.700), hence one per pattern. Off by "
        "default only to leave the integrated confidence unchanged; recommended when fusing 2+ modalities. Needs fit().")
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

        # ---- class priors: validated and normalised once, here ---------------------------------
        K = len(self.labels)

        def prior(name, v, allowed):
            if isinstance(v, str):
                if v in allowed:
                    return v
                raise FusionConfigError(
                    f"{name} must be one of {sorted(allowed)} or {K} positive numbers (label order), got {v!r}")
            try:
                vals = tuple(float(x) for x in v)
            except (TypeError, ValueError):
                raise FusionConfigError(f"{name} must be {K} positive numbers (label order), got {v!r}") from None
            if len(vals) != K:
                raise FusionConfigError(f"{name} needs {K} values (one per label, in label order), got {len(vals)}")
            if not all(math.isfinite(x) and x > 0 for x in vals):
                raise FusionConfigError(
                    f"{name} values must be finite and > 0 (a class with prior 0 could never be predicted): {vals}")
            total = sum(vals)
            return tuple(x / total for x in vals)

        mp: Dict[str, Any] = {}
        for mod, v in dict(self.model_priors).items():
            if mod not in self.modalities:
                raise FusionConfigError(f"model_priors names unknown modality {mod!r}")
            p = prior(f"model_priors[{mod!r}]", v, {"uniform"})
            if not isinstance(p, str) and mod in partial:
                raise FusionConfigError(
                    f"model_priors[{mod!r}]: a non-uniform prior for a partial-support modality is not "
                    "implemented (its conditional distribution over its classes has no defined prior ratio)")
            mp[mod] = p
        set_("model_priors", mp)
        set_("target_prior", prior("target_prior", self.target_prior, {"uniform", "dev"}))
        if not 0.0 <= real("target_prior_power", self.target_prior_power) <= 1.0:
            raise FusionConfigError("target_prior_power must be in [0, 1]")

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "modality_support":
                v = {k: list(x) for k, x in v.items()}
            elif f.name == "model_priors":
                v = {k: (x if isinstance(x, str) else list(x)) for k, x in v.items()}
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
            default = f.default_factory() if f.default_factory is not MISSING else f.default
            rows.append({"name": f.name, "default": default,
                         "basis": f.metadata.get("basis", ""), "doc": f.metadata.get("doc", "")})
        return rows

    @classmethod
    def equal_weight_baseline(cls, **overrides) -> "FusionConfig":
        """Plain average of the available modalities: no calibration, no weighting."""
        from dataclasses import replace
        return replace(cls(), use_calibration=False, use_priors=False, confidence_power=0.0,
                       quality_power=0.0, consistency_power=0.0, pooling="linear", **overrides)
