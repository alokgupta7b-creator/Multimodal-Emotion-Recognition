"""Class-prior correction in pooling (P1): model priors divided out once, target prior applied once.

The exact test is a discrete generative model where the true posterior is known in closed form:
    P(c | x_1..x_M) ~ pi_t(c) * prod_m L_m[c, x_m]          (modalities independent given the class)
while each model reports its own posterior  p_m(c | x_m) ~ pi_m(c) * L_m[c, x_m]  under ITS training prior.
Everything here is synthetic; it checks the algebra, not real-data performance.
"""
import itertools
import warnings

import numpy as np
import pytest

from fusion import FusionConfig, ReliabilityAwareFuser
from fusion.errors import FusionConfigError, FusionInputError

K = 7
MODS = ("text", "audio", "face")


def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / v.sum()


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    L = {m: rng.dirichlet(np.ones(5), size=K) for m in MODS}               # L_m[c, x]
    pi_model = {"text": _norm(rng.dirichlet(np.ones(K) * 2)), "audio": _norm(rng.dirichlet(np.ones(K) * 2)),
                "face": np.full(K, 1.0 / K)}
    pi_target = _norm(rng.dirichlet(np.ones(K) * 2))
    X = {m: rng.integers(0, 5, size=40) for m in MODS}
    post = {m: np.stack([_norm(pi_model[m] * L[m][:, x]) for x in X[m]]) for m in MODS}
    return L, pi_model, pi_target, X, post


def _truth(L, pi_target, X, mods):
    out = []
    for i in range(len(X[MODS[0]])):
        p = pi_target.copy()
        for m in mods:
            p = p * L[m][:, X[m][i]]
        out.append(_norm(p))
    return np.stack(out)


def _fuser(**kw):
    base = dict(use_calibration=False, smoothing=0.0, pooling="log")
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ReliabilityAwareFuser(FusionConfig(**base))


def _run(f, post, mods):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f.fuse_batch({m: post[m] for m in mods}).probs


class TestBackwardCompatibility:
    def test_defaults_are_uniform_and_change_nothing(self, world):
        _, _, _, _, post = world
        a = _run(_fuser(), post, MODS)
        b = _run(_fuser(model_priors={"text": "uniform"}, target_prior="uniform", target_prior_power=0.3), post, MODS)
        np.testing.assert_array_equal(a, b)

    def test_uniform_log_pooling_is_plain_product_of_experts(self, world):
        _, _, _, _, post = world
        got = _run(_fuser(), post, MODS)
        want = np.stack([_norm(np.prod([post[m][i] for m in MODS], axis=0)) for i in range(40)])
        np.testing.assert_allclose(got, want, atol=1e-12)


class TestExactBayes:
    @pytest.mark.parametrize("n_mods", [1, 2, 3])
    def test_matches_true_posterior_for_every_subset(self, world, n_mods):
        L, pi_model, pi_target, X, post = world
        for subset in itertools.combinations(MODS, n_mods):
            f = _fuser(model_priors={m: pi_model[m] for m in MODS}, target_prior=pi_target)
            np.testing.assert_allclose(_run(f, post, subset), _truth(L, pi_target, X, subset), atol=1e-10,
                                       err_msg=f"subset {subset}")

    def test_naive_pooling_is_not_bayes_when_priors_are_not_uniform(self, world):
        """The defect being fixed: without correction the answer depends on how many models are present."""
        L, pi_model, pi_target, X, post = world
        naive = _run(_fuser(), post, ("text", "audio"))
        truth = _truth(L, np.full(K, 1.0 / K), X, ("text", "audio"))
        assert np.abs(naive - truth).max() > 1e-2
        fixed = _run(_fuser(model_priors={m: pi_model[m] for m in MODS}), post, ("text", "audio"))
        np.testing.assert_allclose(fixed, truth, atol=1e-10)

    def test_prior_is_counted_once_for_identical_models(self):
        """Two copies of a model with prior pi and no evidence: output prior is the target, not pi^2."""
        pi = _norm([5, 1, 1, 1, 8, 1, 1])
        post = np.tile(pi, (3, 1))                         # likelihood is flat: posterior == prior
        both = {"text": post, "audio": post}
        out = _run(_fuser(model_priors={"text": pi, "audio": pi}), both, ("text", "audio"))
        np.testing.assert_allclose(out, np.full((3, K), 1.0 / K), atol=1e-12)
        out_t = _run(_fuser(model_priors={"text": pi, "audio": pi}, target_prior=pi), both, ("text", "audio"))
        np.testing.assert_allclose(out_t, np.tile(pi, (3, 1)), atol=1e-12)

    def test_log_pooling_handles_zero_posterior_with_smoothing(self, world):
        _, pi_model, _, _, post = world
        p = {m: post[m].copy() for m in MODS}
        p["text"][0] = np.eye(K)[0]                        # exact zeros elsewhere
        f = _fuser(smoothing=0.01, model_priors={m: pi_model[m] for m in MODS})
        assert np.isfinite(_run(f, p, MODS)).all()


class TestSingleModalityConversion:
    @pytest.mark.parametrize("pooling", ["linear", "log"])
    def test_one_expert_is_converted_exactly(self, world, pooling):
        L, pi_model, pi_target, X, post = world
        f = _fuser(pooling=pooling, model_priors={"text": pi_model["text"]}, target_prior=pi_target)
        np.testing.assert_allclose(_run(f, post, ("text",)), _truth(L, pi_target, X, ("text",)), atol=1e-10)

    def test_linear_outputs_are_distributions(self, world):
        _, pi_model, pi_target, _, post = world
        f = _fuser(pooling="linear", model_priors={m: pi_model[m] for m in MODS}, target_prior=pi_target)
        out = _run(f, post, MODS)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
        assert (out >= 0).all()


class TestTargetPrior:
    def test_dev_target_requires_fit(self, world):
        _, _, _, _, post = world
        f = _fuser(target_prior="dev")
        with pytest.raises(FusionInputError, match="fit"):
            _run(f, post, MODS)

    def test_dev_target_uses_laplace_smoothed_dev_frequencies(self, world):
        _, _, _, _, post = world
        y = np.array([4] * 30 + [3] * 10)
        f = _fuser(target_prior="dev")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.fit({"text": post["text"]}, y)
        want = _norm(np.bincount(y, minlength=K) + 1.0)
        np.testing.assert_allclose(f.resolved_target_prior, want)
        assert (f.resolved_target_prior > 0).all()

    def test_power_zero_is_uniform_and_power_interpolates(self):
        pi = _norm([1, 1, 1, 1, 9, 1, 1])
        assert np.allclose(_fuser(target_prior=pi, target_prior_power=0).resolved_target_prior, 1.0 / K)
        half = _fuser(target_prior=pi, target_prior_power=0.5).resolved_target_prior
        np.testing.assert_allclose(half, _norm(np.sqrt(pi)))
        np.testing.assert_allclose(_fuser(target_prior=pi).resolved_target_prior, pi)

    def test_target_prior_moves_predictions_towards_frequent_class(self):
        post = np.tile(_norm([1, 1, 1, 1, 1.2, 1, 1]), (1, 1))
        flat = _run(_fuser(), {"text": post}, ("text",)).argmax(1)
        skew = _run(_fuser(target_prior=_norm([1, 1, 1, 1, 9, 1, 1])), {"text": post}, ("text",)).argmax(1)
        assert skew[0] == 4 and flat[0] == 4
        post2 = np.tile(_norm([1, 1.1, 1, 1, 1, 1, 1]), (1, 1))
        assert _run(_fuser(), {"text": post2}, ("text",)).argmax(1)[0] == 1
        assert _run(_fuser(target_prior=_norm([1, 1, 1, 1, 9, 1, 1])), {"text": post2}, ("text",)).argmax(1)[0] == 4


class TestBatchSingleParity:
    def test_fuse_equals_fuse_batch(self, world):
        _, pi_model, pi_target, _, post = world
        f = _fuser(model_priors={m: pi_model[m] for m in MODS}, target_prior=pi_target)
        batch = _run(f, post, MODS)
        for i in (0, 7, 39):
            r = f.fuse({m: post[m][i] for m in MODS})
            np.testing.assert_allclose(r.probs, batch[i], atol=1e-12)


class TestPartialSupportUnderTarget:
    def test_partial_modality_still_cannot_move_unsupported_classes(self, world):
        """Policy F with a non-uniform target: text never changes mass on classes it cannot predict."""
        _, pi_model, pi_target, _, post = world
        sup = {"text": ["anger", "fear", "joy", "sadness", "surprise"]}
        mask = np.array([l in sup["text"] for l in FusionConfig().labels])
        rng = np.random.default_rng(3)
        T = np.full((40, K), np.nan)
        T[:, mask] = rng.dirichlet(np.ones(mask.sum()), size=40)
        kw = dict(modality_support=sup, model_priors={"audio": pi_model["audio"], "face": "uniform"},
                  target_prior=pi_target)
        for pooling in ("log", "linear"):
            both = _run(_fuser(pooling=pooling, **kw), {"text": T, "audio": post["audio"], "face": post["face"]}, MODS)
            base = _run(_fuser(pooling=pooling, **kw), {"audio": post["audio"], "face": post["face"]}, MODS[1:])
            np.testing.assert_allclose(both[:, ~mask], base[:, ~mask], atol=1e-12, err_msg=pooling)
            np.testing.assert_allclose(both.sum(axis=1), 1.0, atol=1e-10)

    def test_non_uniform_model_prior_for_partial_modality_is_rejected(self):
        with pytest.raises(FusionConfigError, match="partial"):
            FusionConfig(modality_support={"text": ["anger", "joy"]}, model_priors={"text": _norm(np.arange(1, K + 1))})


class TestConfigValidation:
    @pytest.mark.parametrize("bad", [
        dict(model_priors={"text": [1, 2, 3]}),                       # wrong length
        dict(model_priors={"text": [1, 1, 1, 1, 1, 1, 0]}),           # zero prior
        dict(model_priors={"text": [1, 1, 1, 1, 1, 1, float("nan")]}),
        dict(model_priors={"text": "balanced"}),
        dict(model_priors={"speech": "uniform"}),                     # unknown modality
        dict(target_prior="empirical"),
        dict(target_prior=[1, 2]),
        dict(target_prior=[-1] + [1] * 6),
        dict(target_prior_power=1.5),
        dict(target_prior_power=-0.1),
    ])
    def test_invalid_values_raise(self, bad):
        with pytest.raises(FusionConfigError):
            FusionConfig(**bad)

    def test_priors_are_normalised_and_round_trip(self):
        cfg = FusionConfig(model_priors={"text": [2] * 6 + [4]}, target_prior=[1] * 6 + [3], target_prior_power=0.5)
        assert abs(sum(cfg.model_priors["text"]) - 1) < 1e-12
        assert FusionConfig.from_dict(cfg.to_dict()) == cfg

    def test_defaults_documented(self):
        names = {r["name"]: r for r in FusionConfig.docs()}
        assert names["target_prior"]["default"] == "uniform" and names["target_prior_power"]["default"] == 1.0
        assert names["model_priors"]["default"] == {}


class TestMeldTextAdapter:
    class _Model:
        def __init__(self, classes):
            self.classes_ = np.array(classes)

        def predict_proba(self, texts):
            n = len(self.classes_)
            P = np.tile(np.arange(1, n + 1, dtype=float), (len(texts), 1))
            return P / P.sum(axis=1, keepdims=True)

    def test_reorders_by_name_without_nan(self):
        from fusion import MELD_LABELS, meld_text_probs_from_model
        shuffled = ["sadness", "joy", "neutral", "anger", "surprise", "fear", "disgust"]
        out = meld_text_probs_from_model(self._Model(shuffled), ["a", "b"])
        assert out.shape == (2, 7) and np.isfinite(out).all()
        for j, label in enumerate(MELD_LABELS):                      # column j holds that class's probability
            assert out[0, j] == pytest.approx((shuffled.index(label) + 1) / 28)

    def test_missing_or_unknown_class_raises(self):
        from fusion import meld_text_probs_from_model
        with pytest.raises(ValueError, match="MELD"):
            meld_text_probs_from_model(self._Model(["anger", "fear", "joy", "sadness", "surprise"]), ["a"])
        with pytest.raises(ValueError, match="MELD"):
            meld_text_probs_from_model(self._Model(["love", "anger", "disgust", "fear", "joy", "neutral", "sadness"]), ["a"])
