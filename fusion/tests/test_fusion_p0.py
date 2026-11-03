"""Regression tests for the P0 fixes: validation, batch/single parity, error reporting, the
partial-class-support policy for the text model, and the documented configuration.

Every defect found in the audit has a test here (marked ``Defect``).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from fusion import (
    DAIR_EMOTION_LABELS, DAIR_TO_MELD, MELD_LABELS, TEXT_SUPPORT, FusionConfig, FusionConfigError,
    FusionInputError, ModalityInput, ReliabilityAwareFuser, apply_temperature, equal_weight_fusion,
    fit_temperature, from_ser_prediction, from_text_probs, remap_probs, text_probs_from_model,
    text_support_config,
)
from fusion.calibration import apply_temperature_masked
from fusion.validation import validate_prob_rows

K = len(MELD_LABELS)
IDX = {c: i for i, c in enumerate(MELD_LABELS)}
SUP = [IDX[c] for c in TEXT_SUPPORT]                      # classes the text model can predict
REST = [i for i in range(K) if i not in SUP]              # neutral, disgust
MODS = ("text", "audio", "face")


def vec(top, conf=0.8):
    p = np.full(K, (1 - conf) / (K - 1))
    p[IDX[top]] = conf
    return p


def text_vec(p5):
    """Distribution over the 5 supported classes in MELD order; NaN = class not covered."""
    v = np.full(K, np.nan)
    v[SUP] = p5
    return v


def part_fuser(**kw):
    kw.setdefault("use_calibration", False)
    return ReliabilityAwareFuser(text_support_config(**kw))


# =============================================================================== config
class TestConfig:
    def test_every_option_is_documented_with_a_basis(self):
        rows = FusionConfig.docs()
        assert {r["name"] for r in rows} == set(FusionConfig().to_dict())
        for r in rows:
            assert r["basis"] in ("evidence", "assumption", "engineering"), r["name"]
            assert len(r["doc"]) > 20, r["name"]

    def test_readme_mentions_every_option(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        missing = [r["name"] for r in FusionConfig.docs() if f"`{r['name']}`" not in readme]
        assert not missing, f"README config table is missing {missing}"

    def test_defaults_follow_the_synthetic_ablation(self):
        c = FusionConfig()
        # Per-sample confidence / quality / consistency weighting and priors did not earn their
        # place in the ablation, so they are opt-in. Calibration did, so it is on.
        assert (c.confidence_power, c.quality_power, c.consistency_power) == (0.0, 0.0, 0.0)
        assert c.use_priors is False and c.use_calibration is True
        assert c.pooling == "linear" and c.strict is False

    @pytest.mark.parametrize("kw", [
        dict(pooling="mean"), dict(smoothing=1.0), dict(smoothing=-0.1), dict(confidence_power=-1),
        dict(quality_power=float("nan")), dict(consistency_power=float("inf")), dict(weight_floor=0.0),
        dict(sum_tol=0.0), dict(min_confidence=1.5), dict(conflict_threshold=-1), dict(labels=("a",)),
        dict(labels=("a", "a")), dict(modalities=("text", "text")),
        dict(modality_support={"video": ("joy", "anger")}),
        dict(modality_support={"text": ("joy", "nope")}),
        dict(modality_support={"text": ("joy",)}),
        dict(modality_support={"text": ("joy", "anger"), "audio": ("fear", "joy")}),   # mixed partial sets
    ])
    def test_invalid_config_raises(self, kw):
        with pytest.raises(FusionConfigError):
            FusionConfig(**kw)

    def test_config_error_is_a_value_error(self):
        assert issubclass(FusionConfigError, ValueError) and issubclass(FusionInputError, ValueError)

    def test_roundtrip_and_json(self):
        c = FusionConfig(pooling="log", consistency_power=1.0, modality_support={"text": TEXT_SUPPORT})
        assert FusionConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c

    def test_unknown_option_rejected(self):
        with pytest.raises(FusionConfigError):
            FusionConfig.from_dict({"poolingg": "log"})
        with pytest.raises(TypeError):
            ReliabilityAwareFuser(poolingg="log")

    def test_keyword_overrides_equal_config_object(self):
        a = ReliabilityAwareFuser(pooling="log", consistency_power=1.0)
        b = ReliabilityAwareFuser(FusionConfig(pooling="log", consistency_power=1.0))
        assert a.config == b.config and a.pooling == "log"       # attribute access still works

    def test_equal_weight_baseline_config(self):
        c = FusionConfig.equal_weight_baseline()
        assert (c.use_calibration, c.use_priors, c.pooling) == (False, False, "linear")
        assert ReliabilityAwareFuser(c).method == "equal_weight"
        assert equal_weight_fusion({"text": vec("joy")}).method == "equal_weight"


# ================================================= validation, status codes, strict mode
class TestValidationAndStatus:
    def _matrix(self):
        X = np.tile(vec("joy"), (7, 1))
        X[1] = np.nan                       # missing
        X[2, 0] = np.nan                    # partial NaN
        X[3, 1] = -0.2                      # negative
        X[4] = 0.0                          # zero sum
        X[5] = X[5] * 3.0                   # not normalised
        return X

    def test_batch_reports_a_reason_for_every_row(self):
        b = ReliabilityAwareFuser(use_calibration=False).fuse_batch({"text": self._matrix()})
        assert list(b.status[0]) == ["ok", "missing", "non_finite", "negative", "zero_sum", "renormalized", "ok"]
        assert list(b.available[0]) == [True, False, False, False, False, True, True]
        s = b.summary()
        assert s["status"]["text"]["negative"] == 1 and s["status"]["audio"] == {"missing": 7}
        assert s["no_prediction"] == 4

    def test_defect_batch_path_used_to_hide_reasons(self):
        """Defect D1: all invalid/missing rows looked identical in BatchResult."""
        b = ReliabilityAwareFuser(use_calibration=False).fuse_batch({"text": self._matrix()})
        assert len(set(b.status[0][[1, 2, 3, 4]])) == 4          # four distinct reasons

    def test_quality_problems_are_reported_not_silent(self):
        """Defect D2: NaN / out-of-range quality used to be changed without any trace."""
        P = {"audio": np.tile(vec("joy"), (4, 1))}
        q = {"audio": np.array([0.5, np.nan, 5.0, -2.0])}
        b = ReliabilityAwareFuser(use_calibration=False, quality_power=1.0).fuse_batch(P, q)
        assert list(b.quality_status[1]) == ["ok", "invalid", "clipped", "clipped"]
        assert b.summary()["quality_issues"]["audio"] == {"invalid": 1, "clipped": 2}
        r = ReliabilityAwareFuser(use_calibration=False).fuse(
            {"audio": ModalityInput(vec("joy"), quality=float("nan"))})
        assert r.quality_status["audio"] == "invalid" and any("quality" in w for w in r.warnings)

    def test_strict_mode_raises_with_counts_but_allows_missing(self):
        f = ReliabilityAwareFuser(FusionConfig(strict=True, use_calibration=False))
        X = self._matrix()
        with pytest.raises(FusionInputError, match=r"text: \{.*negative.*\}"):
            f.fuse_batch({"text": X})
        ok = np.tile(vec("joy"), (3, 1))
        ok[1] = np.nan                                           # missing is a legitimate state
        assert f.fuse_batch({"text": ok}).pred.tolist() == [IDX["joy"], -1, IDX["joy"]]
        with pytest.raises(FusionInputError, match="audio quality"):
            f.fuse_batch({"audio": ok[[0, 2]]}, {"audio": np.array([0.5, 7.0])})
        with pytest.raises(FusionInputError):
            f.fuse({"text": np.r_[np.nan, vec("joy")[1:]]})

    def test_interface_errors_raise_clear_exceptions(self):
        f = ReliabilityAwareFuser(use_calibration=False)
        good = np.tile(vec("joy"), (2, 1))
        cases = [
            (lambda: f.fuse({"video": vec("joy")}), "unknown modalities"),
            (lambda: f.fuse({"text": vec("joy")[:5]}), "expected 7"),
            (lambda: f.fuse({"text": ["a"] * K}), "not numeric"),
            (lambda: f.fuse_batch({"text": good, "audio": good[:1]}), "different row counts"),
            (lambda: f.fuse_batch({"text": good[0]}), "shape"),
            (lambda: f.fuse_batch({"text": good[:, :5]}), "shape"),
            (lambda: f.fuse_batch({"text": good}, {"video": np.ones(2)}), "unknown modalities"),
            (lambda: f.fuse_batch({"text": good}, {"audio": np.ones(3)}), "shape"),
        ]
        for call, msg in cases:
            with pytest.raises(FusionInputError, match=msg):
                call()

    def test_all_none_batch_needs_explicit_n(self):
        b = ReliabilityAwareFuser(use_calibration=False).fuse_batch({"text": None}, n=3)
        assert b.pred.tolist() == [-1, -1, -1] and b.uncertain.all()

    def test_empty_batch_is_handled(self):
        b = ReliabilityAwareFuser(use_calibration=False).fuse_batch({"text": np.zeros((0, K))})
        assert b.pred.shape == (0,) and b.probs.shape == (0, K)

    @pytest.mark.parametrize("pooling", ["linear", "log"])
    @pytest.mark.parametrize("partial", [False, True])
    def test_single_and_batch_agree_on_messy_data(self, pooling, partial):
        """One code path: fuse() == fuse_batch() row by row, including invalid rows and quality."""
        rng = np.random.default_rng(7)
        N = 250
        cfg = dict(pooling=pooling, consistency_power=1.0, quality_power=1.0, use_calibration=False)
        f = part_fuser(**cfg) if partial else ReliabilityAwareFuser(**cfg)
        data = {m: rng.dirichlet(np.ones(K), N) for m in MODS}
        if partial:
            data["text"][:, REST] = np.nan
            data["text"][:, SUP] = rng.dirichlet(np.ones(len(SUP)), N)
        for m in MODS:
            idx = rng.choice(N, 40, replace=False)
            data[m][idx[:10]] = np.nan                           # missing
            data[m][idx[10:15], SUP[0]] = np.nan                 # invalid (partial NaN)
            data[m][idx[15:18], SUP[1]] = -0.5                   # negative
        q = {"audio": rng.uniform(-0.5, 1.5, N)}
        q["audio"][:5] = np.nan
        b = f.fuse_batch(data, q)
        covered = {m: (SUP if (partial and m == "text") else list(range(K))) for m in MODS}
        for i in range(N):
            inp = {}
            for m in MODS:
                row = data[m][i]
                missing = np.isnan(row[covered[m]]).all()
                inp[m] = ModalityInput(None if missing else row, quality=q["audio"][i] if m == "audio" else None)
            r = f.fuse(inp)
            if r.label is None:
                assert b.pred[i] == -1 and np.isnan(b.probs[i]).all()
            else:
                assert r.label_index == b.pred[i]
                np.testing.assert_allclose(r.probs, b.probs[i], atol=1e-12, equal_nan=True)
            assert (r.conflict, r.uncertain, r.partial_only) == (
                bool(b.conflict[i]), bool(b.uncertain[i]), bool(b.partial_only[i]))
            for j, m in enumerate(MODS):
                assert r.status[m] == b.status[j, i] and r.quality_status[m] == b.quality_status[j, i]

    def test_defect_no_prediction_rows_do_not_carry_an_invented_distribution(self):
        """Defect D6: a row with no usable modality used to report a uniform 1/K distribution."""
        r = ReliabilityAwareFuser(use_calibration=False).fuse({"text": None})
        assert r.label is None and r.label_index == -1 and r.uncertain and r.confidence == 0.0
        assert np.isnan(r.probs).all()
        b = ReliabilityAwareFuser(use_calibration=False).fuse_batch({"text": np.full((2, K), np.nan)})
        assert np.isnan(b.probs).all() and b.pred.tolist() == [-1, -1]
        assert json.dumps(r.to_dict())                           # NaN serialised as null

    def test_results_are_deterministic(self):
        rng = np.random.default_rng(3)
        data = {m: rng.dirichlet(np.ones(K), 50) for m in MODS}
        f = ReliabilityAwareFuser(pooling="log", consistency_power=1.0).fit(data, rng.integers(0, K, 50))
        a, b = f.fuse_batch(data), f.fuse_batch(data)
        np.testing.assert_array_equal(a.probs, b.probs)
        np.testing.assert_array_equal(a.weights, b.weights)


# ======================================================================== fit() checks
class TestFit:
    def _dev(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, K, n)
        logits = rng.normal(size=(n, K))
        logits[np.arange(n), y] += 1.5
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True), y

    def test_defect_fit_gives_clear_errors(self):
        """Defect D3: fit() used to leak a raw IndexError for bad labels."""
        P, y = self._dev()
        f = lambda: ReliabilityAwareFuser()
        with pytest.raises(FusionInputError, match=r"in \[0, 6\]"):
            f().fit({"text": P}, np.full(len(y), 9))
        with pytest.raises(FusionInputError, match="integers or label names"):
            f().fit({"text": P}, np.full(len(y), 1.5))
        with pytest.raises(FusionInputError, match="not in the label set"):
            f().fit({"text": P}, np.array(["happy"] * len(y)))
        with pytest.raises(FusionInputError, match="row-aligned"):
            f().fit({"text": P}, y[:10])
        with pytest.raises(FusionInputError, match="unknown modalities"):
            f().fit({"video": P}, y)
        with pytest.raises(FusionInputError, match="1-D"):
            f().fit({"text": P}, y.reshape(2, -1))

    def test_label_names_and_indices_are_equivalent(self):
        P, y = self._dev()
        a = ReliabilityAwareFuser().fit({"text": P}, y)
        b = ReliabilityAwareFuser().fit({"text": P}, np.array([MELD_LABELS[i] for i in y]))
        assert a.temperatures["text"] == b.temperatures["text"]

    def test_fit_report_counts_rows(self):
        P, y = self._dev()
        P[:50] = np.nan
        f = ReliabilityAwareFuser().fit({"text": P}, y)
        rep = f.fit_report["text"]
        assert (rep["n_rows"], rep["n_valid"], rep["n_used"]) == (600, 550, 550)
        assert rep["temperature"] == pytest.approx(f.temperatures["text"])

    def test_fit_with_no_usable_rows_keeps_identity_and_says_so(self):
        _, y = self._dev()
        f = ReliabilityAwareFuser().fit({"text": np.full((len(y), K), np.nan)}, y)
        assert f.temperatures["text"] == 1.0 and "no usable dev rows" in f.fit_report["text"]["note"]

    def test_calibration_off_leaves_temperature_at_one(self):
        P, y = self._dev()
        assert ReliabilityAwareFuser(use_calibration=False).fit({"text": P}, y).temperatures["text"] == 1.0

    def test_defect_unfitted_calibration_warns_once(self):
        """Defect D5: the default (calibration on) silently used T=1 if fit() was forgotten."""
        f = ReliabilityAwareFuser()
        with pytest.warns(UserWarning, match=r"fit\(\) has not been called"):
            f.fuse({"text": vec("joy")})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f.fuse({"text": vec("joy")})                         # only once per fuser
            ReliabilityAwareFuser(FusionConfig.equal_weight_baseline()).fuse({"text": vec("joy")})
            P, y = self._dev()
            ReliabilityAwareFuser().fit({"text": P}, y).fuse({"text": vec("joy")})


# ================================================================ partial class support
class TestPartialSupport:
    """The text model (dair-ai/emotion) cannot predict neutral or disgust. Its missing classes must
    never be read as probability 0. See fusion/DESIGN_partial_support.md."""

    def test_text_support_matches_the_label_mapping(self):
        mapped = {DAIR_TO_MELD.get(c, c) for c in DAIR_EMOTION_LABELS}
        assert set(TEXT_SUPPORT) == mapped
        assert set(MELD_LABELS) - set(TEXT_SUPPORT) == {"neutral", "disgust"}
        assert text_support_config().modality_support["text"] == TEXT_SUPPORT

    @pytest.mark.parametrize("pooling", ["linear", "log"])
    def test_text_cannot_change_the_split_between_supported_and_unsupported_classes(self, pooling):
        """Whatever the text says (and however much weight it gets), P(S) and P(neutral, disgust) are
        exactly what the full-support modalities alone give."""
        audio, face = vec("neutral", 0.6), vec("sadness", 0.5)
        # confidence_power=1 makes the text weight depend on how sure the text is.
        f = part_fuser(pooling=pooling, confidence_power=1.0)
        without = f.fuse({"audio": audio, "face": face})
        sure = f.fuse({"text": text_vec([0.01, 0.01, 0.96, 0.01, 0.01]), "audio": audio, "face": face})
        other = f.fuse({"text": text_vec([0.96, 0.01, 0.01, 0.01, 0.01]), "audio": audio, "face": face})
        flat = f.fuse({"text": text_vec([0.21, 0.2, 0.2, 0.2, 0.19]), "audio": audio, "face": face})
        for r in (sure, other, flat):
            assert r.probs.sum() == pytest.approx(1.0) and np.isfinite(r.probs).all()
            assert r.probs[SUP].sum() == pytest.approx(without.probs[SUP].sum())
            np.testing.assert_allclose(r.probs[REST], without.probs[REST])
        assert not np.allclose(sure.probs[SUP], other.probs[SUP])           # text does act inside S

    def test_defect_uninformative_text_must_not_change_the_decision(self):
        """Measured on random audio/face: the first version of the policy changed the arg-max in
        16-19 % of samples when text said nothing (uniform over its classes). In log pooling a
        uniform text is now exactly the identity; in linear pooling it only flattens inside S."""
        rng = np.random.default_rng(0)
        n = 500
        aud, fac = (rng.dirichlet(np.ones(K) * 0.6, n) for _ in range(2))
        uni = np.full((n, K), np.nan)
        uni[:, SUP] = 1.0 / len(SUP)
        for pooling in ("linear", "log"):
            flat = ReliabilityAwareFuser(pooling=pooling, use_calibration=False).fuse_batch(
                {"audio": aud, "face": fac})
            hier = part_fuser(pooling=pooling).fuse_batch({"text": uni, "audio": aud, "face": fac})
            np.testing.assert_allclose(hier.probs[:, REST], flat.probs[:, REST])
            np.testing.assert_allclose(hier.probs[:, SUP].sum(1), flat.probs[:, SUP].sum(1))
            if pooling == "log":
                np.testing.assert_allclose(hier.probs, flat.probs, atol=1e-12)
                assert (hier.pred == flat.pred).all()
            else:                                                    # flattening only inside S
                assert (hier.pred == flat.pred).mean() > 0.8

    def test_text_decides_inside_the_supported_classes(self):
        f = part_fuser(pooling="log")
        flat = vec("joy", 0.2)                                              # nearly uninformative speech
        r = f.fuse({"text": text_vec([0.02, 0.02, 0.9, 0.04, 0.02]), "audio": flat})
        assert r.label == "joy"

    def test_zero_fill_is_a_veto_that_the_policy_avoids(self):
        """Why 0 is wrong: it makes the text model assert 'certainly not neutral'."""
        audio, face = vec("neutral", 0.6), vec("neutral", 0.5)
        p5 = [0.05, 0.05, 0.8, 0.05, 0.05]
        zero_filled = np.zeros(K)
        zero_filled[SUP] = p5
        naive = ReliabilityAwareFuser(pooling="log", use_calibration=False).fuse(
            {"text": zero_filled, "audio": audio, "face": face})
        policy = part_fuser(pooling="log").fuse({"text": text_vec(p5), "audio": audio, "face": face})
        assert naive.probs[IDX["neutral"]] < 0.2 < 0.5 < policy.probs[IDX["neutral"]]
        assert (naive.label, policy.label) == ("joy", "neutral")

    def test_text_only_is_restricted_and_labelled_as_such(self):
        r = part_fuser().fuse({"text": text_vec([0.1, 0.1, 0.6, 0.1, 0.1])})
        assert r.label == "joy" and r.partial_only and r.available == ["text"]
        assert np.isnan(r.probs[REST]).all() and r.probs[SUP].sum() == pytest.approx(1.0)
        assert any("restricted to" in w for w in r.warnings)
        assert r.label in TEXT_SUPPORT                              # can never answer neutral/disgust

    def test_text_with_an_invalid_companion_falls_back_to_text_only(self):
        bad_audio = vec("neutral")
        bad_audio[0] = np.nan
        r = part_fuser().fuse({"text": text_vec([0.1, 0.1, 0.6, 0.1, 0.1]), "audio": bad_audio})
        assert r.partial_only and r.status["audio"] == "non_finite"

    def test_batch_mixes_text_only_both_and_empty_rows(self):
        f = part_fuser(pooling="log")
        t = np.tile(text_vec([0.1, 0.1, 0.6, 0.1, 0.1]), (3, 1))
        a = np.tile(vec("neutral", 0.7), (3, 1))
        a[0] = np.nan                                               # row 0: text only
        t[2] = np.nan
        a[2] = np.nan                                               # row 2: nothing
        b = f.fuse_batch({"text": t, "audio": a})
        assert b.partial_only.tolist() == [True, False, False] and b.pred[2] == -1
        assert np.isnan(b.probs[0][REST]).all() and np.isfinite(b.probs[1]).all()
        assert b.summary()["partial_only"] == 1

    def test_defect_mass_on_an_unsupported_class_is_flagged_not_discarded(self):
        """Defect D4/A: real probability on a class declared unsupported used to vanish silently."""
        contradiction = np.zeros(K)
        contradiction[SUP] = [0.1, 0.1, 0.4, 0.1, 0.1]
        contradiction[IDX["neutral"]] = 0.2
        r = part_fuser().fuse({"text": contradiction, "audio": vec("neutral")})
        assert r.status["text"] == "unsupported_mass" and r.available == ["audio"]
        with pytest.raises(FusionInputError, match="unsupported_mass"):
            part_fuser(strict=True).fuse({"text": contradiction, "audio": vec("neutral")})
        placeholder = np.zeros(K)
        placeholder[SUP] = [0.1, 0.1, 0.4, 0.2, 0.2]
        assert part_fuser().fuse({"text": placeholder, "audio": vec("joy")}).status["text"] == "ok"

    def test_undeclared_support_never_reads_nan_as_zero(self):
        r = ReliabilityAwareFuser(use_calibration=False).fuse(
            {"text": text_vec([0.1, 0.1, 0.6, 0.1, 0.1]), "audio": vec("neutral")})
        assert r.status["text"] == "non_finite" and r.available == ["audio"] and r.label == "neutral"

    def test_masked_temperature_only_touches_supported_columns(self):
        p = np.zeros(K)
        p[SUP] = [0.5, 0.2, 0.1, 0.1, 0.1]
        soft = apply_temperature_masked(p, 3.0, np.isin(np.arange(K), SUP))
        assert soft[REST].tolist() == [0.0, 0.0] and soft.sum() == pytest.approx(1.0)
        assert soft[SUP].max() < p[SUP].max()                       # T > 1 flattens

    def test_fit_uses_only_rows_whose_true_class_is_supported(self):
        rng = np.random.default_rng(11)
        n = 3000
        y = rng.choice(K, n, p=[.11, .03, .03, .17, .47, .07, .12])
        T_true = 2.0
        logits = rng.normal(size=(n, K))
        logits[np.arange(n), y] += 1.0 * np.isin(y, SUP)           # informative only when y is in S
        P = np.full((n, K), np.nan)
        z = logits[:, SUP] * T_true
        e = np.exp(z - z.max(1, keepdims=True))
        P[:, SUP] = e / e.sum(1, keepdims=True)
        f = part_fuser(use_calibration=True).fit({"text": P}, y)
        sup_rows = np.isin(y, SUP)
        pos = {c: i for i, c in enumerate(SUP)}
        expected = fit_temperature(P[sup_rows][:, SUP], np.array([pos[c] for c in y[sup_rows]]))
        assert f.temperatures["text"] == pytest.approx(expected)
        assert f.fit_report["text"]["n_used"] == int(sup_rows.sum()) < n
        assert f.fit_report["text"]["supported_classes"] == list(TEXT_SUPPORT)

    def test_defect_no_measurable_disagreement_without_shared_mass(self):
        """Defect D: the conflict score used a made-up uniform when two models share no mass."""
        f = part_fuser()
        r = f.fuse({"text": text_vec([0, 0, 1, 0, 0]), "audio": np.eye(K)[IDX["neutral"]]})
        assert r.conflict_score == 0.0 and not r.conflict
        r = f.fuse({"text": text_vec([0, 0, 1, 0, 0]), "audio": vec("sadness", 0.9)})
        assert r.conflict and r.conflict_score > 0.5                 # real, measurable disagreement

    def test_numerical_extremes_stay_finite(self):
        for pooling in ("linear", "log"):
            f = ReliabilityAwareFuser(text_support_config(pooling=pooling))      # calibration on, unfitted
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for t in (text_vec([1, 0, 0, 0, 0]), text_vec([1e-320, 1e-320, 1.0, 1e-320, 1e-320]),
                          text_vec([0.2] * 5).astype(np.float32)):
                    r = f.fuse({"text": t, "audio": np.eye(K)[IDX["neutral"]], "face": np.eye(K)[IDX["joy"]]})
                    assert np.isfinite(r.probs).all() and r.probs.sum() == pytest.approx(1.0)

    def test_baseline_can_declare_partial_support(self):
        cfg = FusionConfig.equal_weight_baseline(modality_support={"text": TEXT_SUPPORT})
        r = ReliabilityAwareFuser(cfg).fuse(
            {"text": text_vec([0.1, 0.1, 0.6, 0.1, 0.1]), "audio": vec("neutral", 0.6)})
        assert r.method == "equal_weight" and r.probs.sum() == pytest.approx(1.0)


# ======================================================================= adapters / labels
class TestAdaptersAndLabels:
    def test_remap_fills_uncovered_classes_with_nan_on_request(self):
        out = remap_probs([0.5, 0.5], ["happiness", "sadness"], ["joy", "sadness", "neutral"],
                          {"happiness": "joy"}, missing_fill=np.nan)
        assert out[:2].tolist() == [0.5, 0.5] and np.isnan(out[2])
        out0 = remap_probs([0.5, 0.5], ["happiness", "sadness"], ["joy", "sadness", "neutral"], {"happiness": "joy"})
        assert out0[2] == 0.0                                       # old default kept for compatibility

    def test_remap_dropping_a_label_renormalises_over_covered_classes_only(self):
        out = remap_probs([0.6, 0.2, 0.2], ["a", "b", "c"], ["a", "b", "x"], {"c": None}, missing_fill=np.nan)
        assert out[:2].tolist() == pytest.approx([0.75, 0.25]) and np.isnan(out[2])

    def test_text_adapter_alternative_mapping(self):
        p = np.zeros(6)
        p[DAIR_EMOTION_LABELS.index("love")] = 0.5
        p[DAIR_EMOTION_LABELS.index("joy")] = 0.25
        p[DAIR_EMOTION_LABELS.index("anger")] = 0.25
        out = np.asarray(from_text_probs(p, mapping={"love": None}).probs)       # drop love instead of folding it
        assert out[IDX["joy"]] == pytest.approx(0.5) and out[IDX["anger"]] == pytest.approx(0.5)

    def test_text_probs_from_model_reorders_classes(self):
        class FakeModel:                     # classes_ deliberately shuffled
            classes_ = np.array([3, 0, 5, 1, 4, 2])

            def predict_proba(self, texts):
                base = np.array([[0.4, 0.3, 0.1, 0.1, 0.05, 0.05]] * len(texts))   # columns follow classes_
                return base

        out = text_probs_from_model(FakeModel(), ["x", "y"])
        # classes_ order [3,0,5,1,4,2] -> anger(3)=0.4, sadness(0)=0.3, surprise(5)=0.1, joy(1)=0.1, fear(4)=0.05, love(2)=0.05
        assert out.shape == (2, K) and np.isnan(out[:, IDX["neutral"]]).all()
        assert out[0, IDX["anger"]] == pytest.approx(0.4) and out[0, IDX["joy"]] == pytest.approx(0.15)
        assert np.nansum(out[0]) == pytest.approx(1.0)

        class BadModel(FakeModel):
            classes_ = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="unexpected text model classes_"):
            text_probs_from_model(BadModel(), ["x"])

    def test_ser_adapter_passes_quality_through(self):
        from types import SimpleNamespace
        mi = from_ser_prediction(SimpleNamespace(probability_array=vec("joy")), quality=0.3)
        assert mi.quality == 0.3 and np.asarray(mi.probs).argmax() == IDX["joy"]

    def test_validate_prob_rows_support_columns(self):
        cols = np.isin(np.arange(K), SUP)
        X = np.vstack([text_vec([0.2] * 5), np.where(cols, 0.2, 0.0), np.full(K, np.nan)])
        clean, valid, reason = validate_prob_rows(X, K, columns=cols)
        assert list(reason) == ["ok", "ok", "missing"] and valid.tolist() == [True, True, False]
        np.testing.assert_allclose(clean[0][SUP], 0.2)
        assert clean[0][REST].tolist() == [0.0, 0.0]                # placeholders, not probabilities


# ======================================================================== simulator
class TestSyntheticSimulatorReproducibility:
    def test_same_seed_same_data_different_seed_different_data(self):
        from fusion.experiments import synthetic_experiment as ex
        a, b, c = ex.simulate(300, 5), ex.simulate(300, 5), ex.simulate(300, 6)
        for m in ex.MODS:
            np.testing.assert_array_equal(a["probs"][m], b["probs"][m])
        assert not np.array_equal(a["probs"]["text"], c["probs"]["text"])
        s1, s2 = ex.make_scenario(a, "conflict_30pct", 9), ex.make_scenario(a, "conflict_30pct", 9)
        np.testing.assert_array_equal(s1["probs"]["audio"], s2["probs"]["audio"])

    def test_scenarios_create_the_intended_failures(self):
        from fusion.experiments import synthetic_experiment as ex
        base = ex.simulate(400, 1)
        assert np.isnan(ex.make_scenario(base, "audio_missing", 1)["probs"]["audio"]).all()
        only = ex.make_scenario(base, "text_only", 1)["probs"]
        assert np.isnan(only["audio"]).all() and np.isnan(only["face"]).all() and not np.isnan(only["text"]).any()
        rnd = ex.make_scenario(base, "random_missing_25pct", 1)["probs"]
        missing = np.stack([np.isnan(rnd[m]).all(axis=1) for m in ex.MODS])      # (modality, sample)
        assert missing.any() and not missing.all(axis=0).any()                   # some dropped, never all


# ===================================== findings from the independent review of this module
class TestIndependentReviewFindings:
    """Each test reproduces a defect found by a reviewer who had not seen the implementation."""

    def test_consistency_weighting_must_not_let_text_move_the_base_fusion(self):
        """With consistency_power > 0 the text used to change the audio/face weights, and through them
        P(S), P(neutral), P(disgust) -- which the design says text can never move."""
        rng = np.random.default_rng(5)
        n = 300
        aud, fac = (rng.dirichlet(np.ones(K) * 0.6, n) for _ in range(2))
        txt = np.full((n, K), np.nan)
        txt[:, SUP] = rng.dirichlet(np.ones(len(SUP)) * 0.5, n)
        uni = np.full((n, K), np.nan)
        uni[:, SUP] = 1.0 / len(SUP)
        for pooling in ("linear", "log"):
            f = part_fuser(pooling=pooling, consistency_power=1.0)
            base = f.fuse_batch({"audio": aud, "face": fac})
            with_text = f.fuse_batch({"text": txt, "audio": aud, "face": fac})
            np.testing.assert_allclose(with_text.probs[:, SUP].sum(1), base.probs[:, SUP].sum(1), atol=1e-12)
            np.testing.assert_allclose(with_text.probs[:, REST], base.probs[:, REST], atol=1e-12)
            if pooling == "log":                                   # uniform text = exact identity
                np.testing.assert_allclose(f.fuse_batch({"text": uni, "audio": aud, "face": fac}).probs,
                                           base.probs, atol=1e-12)

    def test_refit_does_not_keep_stale_state(self):
        rng = np.random.default_rng(1)
        n = 400
        y = rng.integers(0, K, n)
        L = rng.normal(size=(n, K))
        L[np.arange(n), y] += 2.0
        P = np.exp(L * 3) / np.exp(L * 3).sum(1, keepdims=True)
        f = ReliabilityAwareFuser(use_priors=True).fit({"text": P, "audio": P, "face": P}, y)
        assert f.temperatures["text"] != 1.0
        f.fit({"audio": P, "face": P}, y)                                # text no longer fitted
        assert f.temperatures["text"] == 1.0 and f.priors["text"] == 1.0
        assert np.mean([f.priors["audio"], f.priors["face"]]) == pytest.approx(1.0)
        f.fit({"audio": P, "face": P, "text": np.full((n, K), np.nan)}, y)   # text present but unusable
        assert f.temperatures["text"] == 1.0 and f.priors["text"] == 1.0

    def test_row_sum_overflow_is_not_a_usable_row(self):
        f = ReliabilityAwareFuser(use_calibration=False)
        r = f.fuse({"audio": [1e308, 1e308, 0, 0, 0, 0, 0], "face": vec("joy")})
        assert r.status["audio"] == "non_finite" and r.available == ["face"]
        assert r.probs.sum() == pytest.approx(1.0)
        assert ReliabilityAwareFuser(use_calibration=False, pooling="log").fuse(
            {"audio": [1e308, 1e308, 0, 0, 0, 0, 0]}).label is None

    def test_tiny_dev_sets_do_not_produce_a_confident_temperature(self):
        f = ReliabilityAwareFuser()
        with pytest.warns(UserWarning, match="min_fit_samples"):
            f.fit({"audio": np.array([vec("joy", 0.6)])}, [IDX["joy"]])
        assert f.temperatures["audio"] == 1.0 and "min_fit_samples" in f.fit_report["audio"]["note"]
        assert ReliabilityAwareFuser(min_fit_samples=1).config.min_fit_samples == 1

    def test_separable_dev_data_flags_a_temperature_at_the_search_bound(self):
        n = 200
        y = np.arange(n) % K
        P = np.eye(K)[y] * 0.999 + 0.001 / K                              # always right, always sure
        with pytest.warns(UserWarning, match="search bound"):
            f = ReliabilityAwareFuser().fit({"audio": P}, y)
        assert f.fit_report["audio"]["temperature_at_bound"] is True

    @pytest.mark.parametrize("kw", [
        dict(sum_tol=float("inf")), dict(sum_tol=0.5), dict(conflict_threshold=float("nan")),
        dict(conflict_threshold=2.0), dict(consistency_power=True), dict(min_fit_samples=0),
        dict(min_fit_samples=True), dict(smoothing=float("nan")),
    ])
    def test_config_validation_holes_are_closed(self, kw):
        with pytest.raises(FusionConfigError):
            FusionConfig(**kw)

    def test_numpy_scalar_options_are_accepted(self):
        FusionConfig(confidence_power=np.float32(1.0), consistency_power=np.int64(1), min_fit_samples=np.int64(10))

    def test_single_sample_quality_matches_batch(self):
        f = ReliabilityAwareFuser(use_calibration=False, quality_power=1.0)
        a = f.fuse({"audio": ModalityInput(vec("joy"), quality=np.array([0.5])), "face": vec("sadness")})
        b = f.fuse({"audio": ModalityInput(vec("joy"), quality=0.5), "face": vec("sadness")})
        assert a.quality_status["audio"] == "ok" and a.weights == b.weights
        for bad in ("high", np.array([0.5, 0.6])):
            with pytest.raises(FusionInputError, match="quality"):
                f.fuse({"audio": ModalityInput(vec("joy"), quality=bad)})
            with pytest.raises(FusionInputError, match="quality"):
                f.fuse_batch({"audio": np.tile(vec("joy"), (2, 1))}, {"audio": bad if isinstance(bad, str) else np.ones((2, 2))})

    def test_unsupported_mass_is_checked_in_total_not_per_column(self):
        row = np.zeros(K)
        row[SUP] = [0.2] * 5
        row[REST] = 0.0006                                                 # each below sum_tol=1e-3, total above
        assert part_fuser().fuse({"text": row, "audio": vec("joy")}).status["text"] == "unsupported_mass"

    def test_text_support_config_defaults_to_log_pooling(self):
        assert text_support_config().pooling == "log"
        assert text_support_config(pooling="linear").pooling == "linear"
