import numpy as np
import pytest

from fusion import (
    MELD_LABELS,
    ModalityInput,
    ReliabilityAwareFuser,
    accuracy,
    apply_temperature,
    confusion_matrix,
    equal_weight_fusion,
    fit_temperature,
    macro_f1,
    remap_probs,
)
from fusion.validation import validate_prob_rows

K = len(MELD_LABELS)
JOY, SAD, NEU = MELD_LABELS.index("joy"), MELD_LABELS.index("sadness"), MELD_LABELS.index("neutral")


def onehot_ish(idx, conf=0.8):
    p = np.full(K, (1 - conf) / (K - 1))
    p[idx] = conf
    return p


def plain_fuser(**kw):
    """Fuser configured to behave as an equal-weight average."""
    base = dict(use_priors=False, use_calibration=False, confidence_power=0,
                quality_power=0, consistency_power=0)
    base.update(kw)
    return ReliabilityAwareFuser(**base)


# ---------------------------------------------------------------- validation
class TestValidation:
    def test_valid_row_ok(self):
        clean, valid, reason = validate_prob_rows(np.array([onehot_ish(JOY)]), K)
        assert valid[0] and reason[0] == "ok"

    def test_nan_negative_zero_missing_flagged(self):
        X = np.array([
            [np.nan] * K,
            [np.nan] + [0.1] * (K - 1),
            [-0.1] + [0.2] * (K - 1),
            [0.0] * K,
        ])
        _, valid, reason = validate_prob_rows(X, K)
        assert not valid.any()
        assert list(reason) == ["missing", "non_finite", "negative", "zero_sum"]

    def test_unnormalised_row_is_renormalised(self):
        clean, valid, reason = validate_prob_rows(np.array([[2.0] + [1.0] * (K - 1)]), K)
        assert valid[0] and reason[0] == "renormalized"
        assert clean[0].sum() == pytest.approx(1.0)

    def test_wrong_class_count_raises(self):
        with pytest.raises(ValueError):
            validate_prob_rows(np.ones((2, K + 1)) / (K + 1), K)

    def test_unknown_modality_raises(self):
        with pytest.raises(ValueError):
            equal_weight_fusion({"video": onehot_ish(JOY)})
        with pytest.raises(ValueError):
            ReliabilityAwareFuser().fuse({"video": onehot_ish(JOY)})

    def test_invalid_modality_is_dropped_not_crashed(self):
        bad = onehot_ish(JOY)
        bad[0] = np.nan
        for r in (equal_weight_fusion({"text": onehot_ish(SAD), "audio": bad}),
                  ReliabilityAwareFuser().fuse({"text": onehot_ish(SAD), "audio": bad})):
            assert r.label == "sadness"
            assert r.available == ["text"] and "audio" in r.missing
            assert any("audio" in w for w in r.warnings)


# ------------------------------------------------------------------ baseline
class TestBaseline:
    def test_matches_brief_example(self):
        # Brief: joy probs 0.80 / 0.35 / 0.75 -> mean 0.633
        def vec(pj):
            p = np.full(K, (1 - pj) / (K - 1))
            p[JOY] = pj
            return p
        r = equal_weight_fusion({"text": vec(0.80), "audio": vec(0.35), "face": vec(0.75)})
        assert r.probs[JOY] == pytest.approx((0.80 + 0.35 + 0.75) / 3)
        assert r.label == "joy"

    def test_missing_modality_averages_the_rest(self):
        a, b = onehot_ish(JOY, 0.7), onehot_ish(NEU, 0.5)
        r = equal_weight_fusion({"text": a, "face": b, "audio": None})
        np.testing.assert_allclose(r.probs, (a + b) / 2)
        assert r.available == ["text", "face"] and r.missing == ["audio"]

    def test_all_missing_returns_no_prediction(self):
        for r in (equal_weight_fusion({}), ReliabilityAwareFuser().fuse({"text": None})):
            assert r.label is None and r.label_index == -1 and r.uncertain
            assert r.confidence == 0.0


# --------------------------------------------------------- reliability fuser
class TestReliabilityFuser:
    def test_equal_config_equals_baseline(self):
        rng = np.random.default_rng(0)
        for pooling in ("linear",):
            for _ in range(20):
                inp = {m: rng.dirichlet(np.ones(K)) for m in ("text", "audio", "face")}
                np.testing.assert_allclose(
                    plain_fuser(pooling=pooling).fuse(inp).probs,
                    equal_weight_fusion(inp).probs, atol=1e-12)

    @pytest.mark.parametrize("pooling", ["linear", "log"])
    def test_output_is_distribution_and_weights_sum_to_one(self, pooling):
        rng = np.random.default_rng(1)
        f = ReliabilityAwareFuser(pooling=pooling, consistency_power=1.0)
        inp = {m: ModalityInput(rng.dirichlet(np.ones(K)), quality=0.7) for m in ("text", "audio", "face")}
        r = f.fuse(inp)
        assert r.probs.sum() == pytest.approx(1.0) and (r.probs >= 0).all()
        assert sum(r.weights.values()) == pytest.approx(1.0)

    def test_missing_modality_has_zero_weight_and_is_listed(self):
        r = ReliabilityAwareFuser().fuse({"text": onehot_ish(JOY), "face": onehot_ish(JOY)})
        assert r.missing == ["audio"] and "audio" not in r.weights
        assert sum(r.weights.values()) == pytest.approx(1.0)

    def test_low_quality_reduces_weight_and_changes_outcome(self):
        text, audio = onehot_ish(JOY, 0.6), onehot_ish(SAD, 0.7)
        f = ReliabilityAwareFuser(confidence_power=0, quality_power=1.0)
        good = f.fuse({"text": text, "audio": ModalityInput(audio, quality=1.0)})
        poor = f.fuse({"text": text, "audio": ModalityInput(audio, quality=0.05)})
        assert poor.weights["audio"] < good.weights["audio"]
        assert good.label == "sadness" and poor.label == "joy"

    def test_quality_out_of_range_is_clipped_with_warning(self):
        r = ReliabilityAwareFuser().fuse(
            {"text": onehot_ish(JOY), "audio": ModalityInput(onehot_ish(JOY), quality=3.0)})
        assert any("clipped" in w for w in r.warnings)
        r = ReliabilityAwareFuser().fuse(
            {"text": onehot_ish(JOY), "audio": ModalityInput(onehot_ish(JOY), quality=float("nan"))})
        assert any("ignored" in w for w in r.warnings)

    def test_confidence_weighting_favours_the_sure_modality(self):
        sure, unsure = onehot_ish(JOY, 0.95), np.full(K, 1.0 / K)
        unsure[SAD] += 0.02
        unsure /= unsure.sum()
        r = ReliabilityAwareFuser(confidence_power=2.0).fuse({"text": sure, "audio": unsure})
        assert r.weights["text"] > 0.9

    def test_conflict_is_flagged_not_hidden(self):
        r = ReliabilityAwareFuser().fuse({"text": onehot_ish(JOY, 0.9), "audio": onehot_ish(SAD, 0.9)})
        assert r.conflict and r.uncertain and r.conflict_score > 0.25

    def test_agreement_is_not_a_conflict(self):
        r = ReliabilityAwareFuser().fuse({"text": onehot_ish(JOY, 0.9), "audio": onehot_ish(JOY, 0.8)})
        assert not r.conflict

    def test_single_modality_never_a_conflict(self):
        r = ReliabilityAwareFuser().fuse({"text": onehot_ish(JOY)})
        assert not r.conflict and r.conflict_score == 0.0 and r.label == "joy"

    def test_consistency_downweights_the_outlier(self):
        inp = {"text": onehot_ish(JOY, 0.8), "audio": onehot_ish(JOY, 0.8), "face": onehot_ish(SAD, 0.8)}
        off = plain_fuser(consistency_power=0.0).fuse(inp)
        on = plain_fuser(consistency_power=2.0).fuse(inp)
        assert on.weights["face"] < off.weights["face"]

    def test_min_confidence_marks_uncertain(self):
        r = ReliabilityAwareFuser(min_confidence=0.95).fuse({"text": onehot_ish(JOY, 0.6)})
        assert r.uncertain and r.label == "joy"

    def test_batch_matches_single(self):
        rng = np.random.default_rng(3)
        N = 40
        data = {m: rng.dirichlet(np.ones(K), N) for m in ("text", "audio", "face")}
        data["audio"][::5] = np.nan
        data["face"][::7] = np.nan
        qual = {"audio": rng.uniform(0, 1, N)}
        f = ReliabilityAwareFuser(pooling="log", consistency_power=1.0)
        b = f.fuse_batch(data, qual)
        for i in range(N):
            inp = {m: ModalityInput(None if np.isnan(data[m][i]).all() else data[m][i],
                                    quality=qual["audio"][i] if m == "audio" else None)
                   for m in data}
            r = f.fuse(inp)
            np.testing.assert_allclose(r.probs, b.probs[i], atol=1e-10)
            assert r.label_index == b.pred[i] and r.conflict == b.conflict[i]

    def test_batch_all_missing_row_gives_minus_one(self):
        data = {m: np.full((2, K), np.nan) for m in ("text", "audio", "face")}
        data["text"][0] = onehot_ish(JOY)
        b = ReliabilityAwareFuser().fuse_batch(data)
        assert list(b.pred) == [JOY, -1] and b.uncertain[1]


# --------------------------------------------------------------- fit / calib
class TestFitAndCalibration:
    def _overconfident_dev(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, K, n)
        logits = rng.normal(size=(n, K))
        logits[np.arange(n), y] += 1.2
        return apply_temperature(np.exp(logits) / np.exp(logits).sum(1, keepdims=True), 0.25), y

    def test_temperature_softens_overconfident_model(self):
        P, y = self._overconfident_dev()
        assert fit_temperature(P, y) > 2.0

    def test_temperature_near_one_when_already_calibrated(self):
        rng = np.random.default_rng(1)
        n = 6000
        P = rng.dirichlet(np.ones(K) * 0.7, n)
        y = np.array([rng.choice(K, p=p) for p in P])  # labels drawn from P itself
        assert fit_temperature(P, y) == pytest.approx(1.0, abs=0.15)

    def test_fit_sets_temperature_and_priors(self):
        P, y = self._overconfident_dev()
        f = ReliabilityAwareFuser().fit({"text": P, "audio": P}, y)
        assert f.fitted and f.temperatures["text"] > 2.0
        assert f.priors["text"] > 0 and np.mean([f.priors["text"], f.priors["audio"]]) == pytest.approx(1.0)

    def test_fit_ignores_missing_rows(self):
        P, y = self._overconfident_dev()
        P2 = P.copy()
        P2[::2] = np.nan
        f = ReliabilityAwareFuser().fit({"text": P2}, y)
        assert f.temperatures["text"] > 2.0


# ------------------------------------------------------------------- metrics
class TestMetricsAndLabels:
    def test_confusion_and_macro_f1_by_hand(self):
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        cm = confusion_matrix(y_true, y_pred, 3)
        np.testing.assert_array_equal(cm, [[2, 1, 0], [0, 1, 1], [0, 0, 1]])
        # class0: P=1,R=2/3,F1=.8 ; class1: P=.5,R=.5,F1=.5 ; class2: P=.5,R=1,F1=2/3
        assert macro_f1(y_true, y_pred, 3) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
        assert accuracy(y_true, y_pred) == pytest.approx(4 / 6)

    def test_no_prediction_counts_as_wrong(self):
        assert accuracy(np.array([1, 1]), np.array([1, -1])) == 0.5

    def test_remap_probs_merges_and_drops(self):
        src = ["happiness", "sadness", "other"]
        out = remap_probs([0.5, 0.3, 0.2], src, ["joy", "sadness"], {"happiness": "joy", "other": None})
        np.testing.assert_allclose(out, [0.5 / 0.8, 0.3 / 0.8])

    def test_remap_unknown_destination_raises(self):
        with pytest.raises(ValueError):
            remap_probs([1.0], ["x"], ["joy"], {"x": "nope"})


# ------------------------------------------------------------------ adapters
class TestAdapters:
    def test_ser_prediction_adapter_is_duck_typed(self):
        from types import SimpleNamespace
        from fusion import from_ser_prediction
        pred = SimpleNamespace(probability_array=onehot_ish(JOY, 0.7))
        mi = from_ser_prediction(pred, quality=0.4)
        assert mi.quality == 0.4 and np.asarray(mi.probs).argmax() == JOY

    def test_text_adapter_maps_dair_labels_to_meld(self):
        from fusion import DAIR_EMOTION_LABELS, from_text_probs
        p = np.zeros(6)
        p[DAIR_EMOTION_LABELS.index("love")] = 0.6
        p[DAIR_EMOTION_LABELS.index("joy")] = 0.2
        p[DAIR_EMOTION_LABELS.index("anger")] = 0.2
        out = np.asarray(from_text_probs(p).probs)
        assert out.shape == (K,) and out.sum() == pytest.approx(1.0)
        assert out[JOY] == pytest.approx(0.8)                       # love folded into joy
        assert out[NEU] == 0.0 and out[MELD_LABELS.index("disgust")] == 0.0


class TestSmoothing:
    def test_zero_probability_class_is_not_an_absolute_veto(self):
        text = np.zeros(K); text[JOY] = 0.5; text[SAD] = 0.5            # model has no neutral class
        audio = onehot_ish(NEU, 0.9)
        face = onehot_ish(NEU, 0.9)
        r = ReliabilityAwareFuser(pooling="log", use_priors=False, use_calibration=False).fuse(
            {"text": text, "audio": audio, "face": face})
        assert r.probs[NEU] > 0 and r.label == "neutral"

    def test_invalid_smoothing_rejected(self):
        with pytest.raises(ValueError):
            ReliabilityAwareFuser(smoothing=1.0)
