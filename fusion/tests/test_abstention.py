"""Leakage-resistant choice of the abstention threshold. Synthetic inputs: checks logic, not real performance."""
import warnings

import numpy as np
import pytest

from fusion import (FusionConfig, ReliabilityAwareFuser, grouped_folds, out_of_fold_confidence,
                    select_min_confidence)
from fusion.errors import FusionInputError

K = 7


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    groups = np.repeat(np.arange(n // 6), 6)
    P = {}
    for m, s in (("text", 1.5), ("audio", 2.5)):
        z = rng.normal(size=(n, K)) * s
        z[np.arange(n), y] += 1.8
        e = np.exp(z - z.max(1, keepdims=True))
        P[m] = e / e.sum(1, keepdims=True)
    return P, y, groups


class TestGroupedFolds:
    def test_every_group_in_exactly_one_fold_and_all_folds_used(self):
        g = np.repeat(np.arange(40), 5)
        f = grouped_folds(g, 5, seed=3)
        for u in np.unique(g):
            assert len(set(f[g == u])) == 1
        assert set(f) == set(range(5))

    def test_deterministic_and_seed_dependent(self):
        g = np.repeat(np.arange(40), 5)
        np.testing.assert_array_equal(grouped_folds(g, 5, 1), grouped_folds(g, 5, 1))
        assert (grouped_folds(g, 5, 1) != grouped_folds(g, 5, 2)).any()

    def test_string_group_ids_and_bad_arguments(self):
        assert len(grouped_folds(np.array(["a", "b", "c", "a"]), 3)) == 4
        with pytest.raises(FusionInputError):
            grouped_folds(np.arange(3), 5)             # more folds than groups
        with pytest.raises(FusionInputError):
            grouped_folds(np.arange(10), 1)
        with pytest.raises(FusionInputError):
            grouped_folds(np.zeros((3, 2)), 2)


class TestOutOfFold:
    def test_scores_equal_a_manual_refit_that_excludes_the_fold(self):
        P, y, g = _data()
        cfg = FusionConfig(pooling="log")
        oof = out_of_fold_confidence(cfg, P, y, g, n_folds=4, seed=2)
        fold = grouped_folds(g, 4, 2)
        k = 1
        tr, te = fold != k, fold == k
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = ReliabilityAwareFuser(cfg).fit({m: p[tr] for m, p in P.items()}, y[tr])
            r = f.fuse_batch({m: p[te] for m, p in P.items()})
        np.testing.assert_allclose(oof["confidence"][te], r.confidence)
        np.testing.assert_array_equal(oof["pred"][te], r.pred)

    def test_held_out_labels_cannot_influence_their_own_score(self):
        """Flip every label inside one fold: that fold's OOF scores must not change (only other folds' do)."""
        P, y, g = _data()
        cfg = FusionConfig()
        base = out_of_fold_confidence(cfg, P, y, g, 4, 0)
        fold = base["fold"]
        y2 = y.copy()
        y2[fold == 2] = (y2[fold == 2] + 1) % K
        alt = out_of_fold_confidence(cfg, P, y2, g, 4, 0)
        np.testing.assert_array_equal(base["confidence"][fold == 2], alt["confidence"][fold == 2])
        assert (base["confidence"][fold != 2] != alt["confidence"][fold != 2]).any()

    def test_rows_without_any_modality_are_never_correct(self):
        P, y, g = _data()
        P = {m: p.copy() for m, p in P.items()}
        for p in P.values():
            p[:6] = np.nan                                  # first dialogue has no usable modality
        oof = out_of_fold_confidence(FusionConfig(), P, y, g, 4, 0)
        assert (oof["pred"][:6] == -1).all() and not oof["correct"][:6].any()

    def test_row_count_mismatch_raises(self):
        P, y, g = _data()
        with pytest.raises(FusionInputError):
            out_of_fold_confidence(FusionConfig(), P, y[:-1], g[:-1], 4, 0)


class TestSelectThreshold:
    def test_coverage_target_and_risk_goes_down(self):
        rng = np.random.default_rng(0)
        conf = rng.uniform(0.2, 1, 5000)
        ok = rng.uniform(size=5000) < conf                    # confidence is informative
        thr, info = select_min_confidence(conf, ok, coverage=0.5)
        assert abs(info["coverage"] - 0.5) < 0.01
        assert info["risk"] < 1 - info["accuracy_all"]
        assert (conf >= thr).mean() == pytest.approx(info["coverage"])

    def test_max_risk_gives_highest_coverage_meeting_it(self):
        rng = np.random.default_rng(1)
        conf = rng.uniform(0.2, 1, 5000)
        ok = rng.uniform(size=5000) < conf
        thr, info = select_min_confidence(conf, ok, max_risk=0.15)
        assert info["feasible"] and info["risk"] <= 0.15 + 1e-9
        lower = conf >= np.sort(conf)[np.searchsorted(np.sort(conf), thr) - 1]   # one step lower keeps one more row
        assert (~ok[lower]).mean() > 0.15 or lower.mean() == info["coverage"]

    def test_infeasible_risk_abstains_on_everything(self):
        conf = np.array([0.9, 0.8, 0.7])
        thr, info = select_min_confidence(conf, np.array([False, False, False]), max_risk=0.1)
        assert not info["feasible"] and info["coverage"] == 0.0 and (conf < thr).all()

    def test_ties_are_kept_together(self):
        conf = np.array([0.5] * 10)
        _, info = select_min_confidence(conf, np.array([True] * 5 + [False] * 5), coverage=0.3)
        assert info["coverage"] == 1.0

    def test_argument_validation(self):
        c, ok = np.array([0.5, 0.6]), np.array([True, False])
        for kw in (dict(), dict(coverage=0.5, max_risk=0.1), dict(coverage=0.0), dict(coverage=1.2), dict(max_risk=-1)):
            with pytest.raises(FusionInputError):
                select_min_confidence(c, ok, **kw)
        with pytest.raises(FusionInputError):
            select_min_confidence(c, ok[:1], coverage=0.5)
        with pytest.raises(FusionInputError):
            select_min_confidence(np.array([]), np.array([], dtype=bool), coverage=0.5)

    def test_threshold_is_usable_as_min_confidence(self):
        P, y, g = _data()
        oof = out_of_fold_confidence(FusionConfig(), P, y, g, 4, 0)
        thr, _ = select_min_confidence(oof["confidence"], oof["correct"], coverage=0.8)
        # conflict_threshold=1.0 switches the (provisional) conflict flag off so only the confidence rule is tested
        cfg = FusionConfig(min_confidence=min(thr, 1.0), conflict_threshold=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ReliabilityAwareFuser(cfg).fit(P, y).fuse_batch(P)
        assert 0.6 < (~r.uncertain).mean() < 0.95
