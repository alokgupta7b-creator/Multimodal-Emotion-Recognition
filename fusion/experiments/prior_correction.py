"""Class-prior correction in log pooling, on SIMULATED data with an exactly known posterior.

NOT evidence about real MELD performance: no real audio/face predictions exist yet. This shows only
what the algebra implies, and where a wrong declaration hurts.

Simulator: x_m | c ~ N(mu_m[c], sigma_m^2 I) per modality, independent given the class. Model m reports
its EXACT posterior under its own training prior pi_m (so no estimation error, no calibration issue).
Test labels follow the MELD train class frequencies (neutral about 47 %). Because the true posterior is
known, "oracle" is the best any fusion could do with these inputs.

Methods (log pooling, calibration off, equal weights, so only the prior handling differs):
  naive            no model_priors, uniform target  (the behaviour before this change)
  declared+uniform model_priors declared correctly, uniform target
  declared+natural model_priors declared correctly, target = natural class frequencies
  misdeclared      the opposite of the truth declared (unweighted models declared uniform, balanced declared as pi)
  oracle           exact Bayes posterior with the matching target

    python -m fusion.experiments.prior_correction     # a few seconds, writes fusion/results/prior_correction.json
"""
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import itertools  # noqa: E402

import numpy as np  # noqa: E402

from fusion import FusionConfig, MELD_LABELS, ReliabilityAwareFuser, accuracy, macro_f1  # noqa: E402

K, D, N, SEEDS = 7, 6, 20000, range(5)
MODS = ("text", "audio", "face")
COUNTS = dict(anger=1109, disgust=271, fear=268, joy=1743, neutral=4710, sadness=683, surprise=1205)
PI = np.array([COUNTS[l] for l in MELD_LABELS], float)
PI /= PI.sum()
UNIF = np.full(K, 1.0 / K)
SIGMA = {"text": 1.6, "audio": 2.0, "face": 1.8}          # different strengths


def world(seed):
    rng = np.random.default_rng(seed)
    mu = {m: rng.normal(size=(K, D)) for m in MODS}
    y = rng.choice(K, size=N, p=PI)
    x = {m: mu[m][y] + SIGMA[m] * rng.normal(size=(N, D)) for m in MODS}
    return mu, y, x


def loglik(mu, m, x):
    d2 = ((x[:, None, :] - mu[m][None]) ** 2).sum(-1)
    return -d2 / (2 * SIGMA[m] ** 2)


def posterior(ll, prior):
    z = ll + np.log(prior)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fuse(post, mods, declared, target):
    cfg = FusionConfig(pooling="log", use_calibration=False, smoothing=0.0,
                       model_priors=declared, target_prior=target)
    return ReliabilityAwareFuser(cfg).fuse_batch({m: post[m] for m in mods}).probs


def score(P, y):
    pred = P.argmax(1)
    return dict(acc=accuracy(y, pred), macro_f1=macro_f1(y, pred, K), neutral_share=float((pred == 4).mean()))


SCENARIOS = {  # the prior each model was TRAINED with
    "all balanced (class_weight=balanced)": {"text": UNIF, "audio": UNIF, "face": UNIF},
    "all unweighted (natural prior)": {"text": PI, "audio": PI, "face": PI},
    "mixed: text unweighted, audio/face balanced": {"text": PI, "audio": UNIF, "face": UNIF},
}


def declared_of(train):
    return {m: ("uniform" if np.allclose(p, UNIF) else p) for m, p in train.items()}


def flipped(train):
    return {m: (PI if np.allclose(p, UNIF) else "uniform") for m, p in train.items()}


def main():
    out, worst_oracle_gap = {}, 0.0
    for sname, train in SCENARIOS.items():
        acc_rows = {}
        for n_mod in (1, 2, 3):
            agg = {}
            for seed in SEEDS:
                mu, y, x = world(seed)
                ll = {m: loglik(mu, m, x[m]) for m in MODS}
                post = {m: posterior(ll[m], train[m]) for m in MODS}
                for subset in itertools.combinations(MODS, n_mod):
                    truth_u = posterior(sum(ll[m] for m in subset), UNIF)
                    truth_n = posterior(sum(ll[m] for m in subset), PI)
                    got_u = fuse(post, subset, declared_of(train), "uniform")
                    got_n = fuse(post, subset, declared_of(train), PI)
                    worst_oracle_gap = max(worst_oracle_gap, np.abs(got_u - truth_u).max(),
                                           np.abs(got_n - truth_n).max())
                    res = {
                        "naive": score(fuse(post, subset, {}, "uniform"), y),
                        "declared+uniform": score(got_u, y),
                        "declared+natural": score(got_n, y),
                        "misdeclared": score(fuse(post, subset, flipped(train), "uniform"), y),
                        "oracle uniform target": score(truth_u, y),
                        "oracle natural target": score(truth_n, y),
                    }
                    for k, v in res.items():
                        for met, val in v.items():
                            agg.setdefault(k, {}).setdefault(met, []).append(val)
            acc_rows[f"{n_mod} modalit{'y' if n_mod == 1 else 'ies'} available"] = {
                k: {met: round(float(np.mean(v)), 4) for met, v in d.items()} for k, d in agg.items()}
        out[sname] = acc_rows
    out["_max_abs_gap_corrected_vs_exact_bayes"] = float(worst_oracle_gap)
    out["_note"] = ("SIMULATED. Gaussian class-conditionals, exact per-model posteriors, test labels at the MELD train "
                    f"class frequencies, mean over {len(list(SEEDS))} seeds and all modality subsets of each size.")
    p = Path(__file__).resolve().parents[1] / "results" / "prior_correction.json"
    p.write_text(json.dumps(out, indent=2))
    print("SIMULATED results (not real-data performance); macro-F1 / accuracy / predicted-neutral share")
    for sname, rows in out.items():
        if sname.startswith("_"):
            continue
        print(f"\n{sname}")
        for nm, tab in rows.items():
            print(f"  {nm}")
            for meth, v in tab.items():
                print(f"    {meth:24s} macroF1 {v['macro_f1']:.3f}  acc {v['acc']:.3f}  neutral {v['neutral_share']:.3f}")
    print(f"\nmax |corrected - exact Bayes| over all runs: {worst_oracle_gap:.2e}")


if __name__ == "__main__":
    main()
