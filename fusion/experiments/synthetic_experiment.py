"""Synthetic stress-test of the fusion module.

IMPORTANT: this uses *simulated* model outputs. It checks that the fusion
mechanics behave as designed (calibration, weighting, missing/conflict
handling, no tuning on test). It says nothing about real accuracy on MELD or
CMU-MOSEI -- re-run the same protocol with the real text/audio/face
predictions once the teammates' models are available.

Protocol
--------
* dev  split (seed 1): fit temperatures + priors on clean data.
* dev-stress split: dev data pushed through the same failure scenarios; the
  fusion hyper-parameters are chosen here (never on test).
* test split (seed 2): one frozen configuration per method is reported.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import (  # noqa: E402
    MELD_LABELS, ReliabilityAwareFuser, accuracy, expected_calibration_error, macro_f1,
)

K = len(MELD_LABELS)
MODS = ("text", "audio", "face")
# MELD-like skew (approximate, for realism of macro-F1 only).
CLASS_PRIOR = np.array([0.11, 0.03, 0.03, 0.17, 0.47, 0.07, 0.12])
CLASS_PRIOR = CLASS_PRIOR / CLASS_PRIOR.sum()

# Per-modality simulation: signal strength, over-confidence (probs = softmax(scale*logits)).
STRENGTH = {"text": 2.4, "audio": 1.9, "face": 1.6}
OVERCONF = {"text": 1.3, "audio": 2.2, "face": 1.8}


def softmax(L):
    L = L - L.max(-1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(-1, keepdims=True)


def simulate(n: int, seed: int) -> dict:
    """Clean, naturally-varying data: latent quality q controls each modality's signal."""
    rng = np.random.default_rng(seed)
    y = rng.choice(K, size=n, p=CLASS_PRIOR)
    probs, qual = {}, {}
    for m in MODS:
        q = rng.beta(5, 2, n)
        logits = STRENGTH[m] * q[:, None] * np.eye(K)[y] + rng.normal(size=(n, K))
        probs[m] = softmax(OVERCONF[m] * logits)
        # What a quality estimator would report: a noisy view of q (text has none).
        qual[m] = None if m == "text" else np.clip(q + rng.normal(0, 0.08, n), 0, 1)
    return {"y": y, "probs": probs, "qual": qual, "q": None}


def make_scenario(base: dict, name: str, seed: int) -> dict:
    """Apply a failure scenario to a copy of ``base``."""
    rng = np.random.default_rng(seed)
    n = len(base["y"])
    y = base["y"]
    probs = {m: p.copy() for m, p in base["probs"].items()}
    qual = {m: (None if q is None else q.copy()) for m, q in base["qual"].items()}

    def drop(m):
        probs[m][:] = np.nan

    def degrade_audio(report_quality: bool):
        # Heavy noise: signal mostly destroyed, model stays over-confident.
        logits = 0.35 * STRENGTH["audio"] * np.eye(K)[y] * rng.beta(5, 2, n)[:, None] + rng.normal(size=(n, K))
        probs["audio"] = softmax(OVERCONF["audio"] * logits)
        qual["audio"] = np.clip(rng.beta(2, 5, n), 0, 1) if report_quality else None

    if name == "clean":
        pass
    elif name == "audio_noisy_quality_known":
        degrade_audio(True)
    elif name == "audio_noisy_quality_unknown":
        degrade_audio(False)
    elif name == "audio_missing":
        drop("audio")
    elif name == "face_missing":
        drop("face")
    elif name == "text_missing":
        drop("text")
    elif name == "text_only":
        drop("audio"); drop("face")
    elif name == "random_missing_25pct":
        for m in MODS:
            probs[m][rng.random(n) < 0.25] = np.nan
        none = np.all([np.isnan(probs[m]).all(1) for m in MODS], axis=0)
        probs["text"][none] = base["probs"]["text"][none]  # keep >=1 modality per sample
    elif name == "conflict_30pct":
        # In 30% of samples one random modality confidently asserts a wrong class
        # (and its reported quality stays high -- the model does not know it is wrong).
        hit = rng.random(n) < 0.30
        who = rng.integers(0, 3, n)
        wrong = (y + rng.integers(1, K, n)) % K
        for i, m in enumerate(MODS):
            sel = hit & (who == i)
            p = np.full((sel.sum(), K), 0.1 / (K - 1))
            p[np.arange(sel.sum()), wrong[sel]] = 0.9
            probs[m][sel] = p
    else:
        raise ValueError(name)
    return {"y": y, "probs": probs, "qual": qual}


SCENARIOS = [
    "clean", "audio_noisy_quality_known", "audio_noisy_quality_unknown",
    "audio_missing", "face_missing", "text_missing", "text_only",
    "random_missing_25pct", "conflict_30pct",
]


def concat(parts):
    y = np.concatenate([p["y"] for p in parts])
    probs = {m: np.concatenate([p["probs"][m] for p in parts]) for m in MODS}
    qual = {}
    for m in MODS:
        if all(p["qual"][m] is None for p in parts):
            qual[m] = None
        else:  # unknown quality -> 1.0
            qual[m] = np.concatenate([np.ones(len(p["y"])) if p["qual"][m] is None else p["qual"][m] for p in parts])
    return {"y": y, "probs": probs, "qual": qual}


# ----------------------------------------------------------------- evaluation
def unimodal_pred(data, m):
    p = data["probs"][m]
    miss = np.isnan(p).any(1)
    pred = np.where(miss, -1, np.nan_to_num(p).argmax(1))
    return pred


def evaluate(pred, y):
    return accuracy(y, pred), macro_f1(y, pred, K)


def run_fuser(f, data):
    b = f.fuse_batch(data["probs"], data["qual"])
    return b


def make_fuser(cfg, dev):
    f = ReliabilityAwareFuser(**cfg)
    return f.fit(dev["probs"], dev["y"]) if cfg.get("use_calibration", True) or cfg.get("use_priors", True) else f


BASELINE_CFG = dict(use_priors=False, use_calibration=False, confidence_power=0,
                    quality_power=0, consistency_power=0, pooling="linear")
CALIB_ONLY_CFG = dict(use_priors=False, use_calibration=True, confidence_power=0,
                      quality_power=0, consistency_power=0, pooling="linear")


def tune(dev, dev_stress):
    """Pick (confidence, quality, consistency, pooling) on dev-stress by macro-F1."""
    results = []
    for pooling, g, b, l in itertools.product(["linear", "log"], [0, 1, 2], [0, 1, 2], [0, 1, 2, 4]):
        cfg = dict(pooling=pooling, confidence_power=g, quality_power=b, consistency_power=l)
        f = make_fuser(cfg, dev)
        pred = run_fuser(f, dev_stress).pred
        results.append((macro_f1(dev_stress["y"], pred, K), cfg))
    results.sort(key=lambda t: -t[0])
    best = {}
    for pooling in ("linear", "log"):
        best[pooling] = next(c for s, c in results if c["pooling"] == pooling)
    return best, results


def auroc(score, is_pos):
    """P(score of a random positive > score of a random negative); ties count half."""
    score = np.asarray(score, float)
    is_pos = np.asarray(is_pos, bool)
    n_pos, n_neg = int(is_pos.sum()), int((~is_pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    uniq, inv = np.unique(score, return_inverse=True)
    counts = np.bincount(inv)
    cum = np.cumsum(counts)
    avg_rank = (cum - counts + 1 + cum) / 2.0          # average rank per unique value
    ranks = avg_rank[inv]
    u = ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


ABLATIONS = {  # all on calibrated, linear pooling; each row ADDS to the previous "calib only"
    "calib_only": dict(use_priors=False, confidence_power=0, quality_power=0, consistency_power=0),
    "+priors": dict(use_priors=True, confidence_power=0, quality_power=0, consistency_power=0),
    "+confidence": dict(use_priors=False, confidence_power=1, quality_power=0, consistency_power=0),
    "+quality": dict(use_priors=False, confidence_power=0, quality_power=1, consistency_power=0),
    "+consistency": dict(use_priors=False, confidence_power=0, quality_power=0, consistency_power=1),
    "all_three": dict(use_priors=True, confidence_power=1, quality_power=1, consistency_power=1),
}
MAIN = ["equal_weight", "equal+calibration", "reliability_linear", "reliability_log"]


def run_once(seed, n_dev, n_test, verbose):
    dev = simulate(n_dev, seed=10 * seed + 1)
    test = simulate(n_test, seed=10 * seed + 2)
    dev_stress = concat([make_scenario(dev, s, 100 * seed + i) for i, s in enumerate(SCENARIOS)])
    best, _ = tune(dev, dev_stress)
    methods = {
        "equal_weight": make_fuser(BASELINE_CFG, dev),
        "equal+calibration": make_fuser(CALIB_ONLY_CFG, dev),
        "reliability_linear": make_fuser(best["linear"], dev),
        "reliability_log": make_fuser(best["log"], dev),
    }
    abl = {k: make_fuser(dict(pooling="linear", **v), dev) for k, v in ABLATIONS.items()}
    abl["all_three_log"] = make_fuser(dict(pooling="log", **ABLATIONS["all_three"]), dev)

    scen = {s: make_scenario(test, s, 1000 * seed + i) for i, s in enumerate(SCENARIOS)}
    f1, acc = {}, {}
    for s, data in scen.items():
        for m in MODS:
            pred = unimodal_pred(data, m)
            ok = not (pred == -1).all()
            acc[(s, f"{m}_only")], f1[(s, f"{m}_only")] = evaluate(pred, data["y"]) if ok else (np.nan, np.nan)
        for name, f in {**methods, **abl}.items():
            a, m_ = evaluate(run_fuser(f, data).pred, data["y"])
            acc[(s, name)], f1[(s, name)] = a, m_

    if verbose:
        print("Chosen on dev-stress (never on test):")
        for k, c in best.items():
            print(f"  {k:6s}", {a: c[a] for a in ("confidence_power", "quality_power", "consistency_power")})
        from fusion import apply_temperature
        print("\nCalibration on dev (ECE before -> after temperature scaling):")
        fc = methods["equal+calibration"]
        for m in MODS:
            b4 = expected_calibration_error(dev["probs"][m], dev["y"])
            af = expected_calibration_error(apply_temperature(dev["probs"][m], fc.temperatures[m]), dev["y"])
            print(f"  {m:5s} T={fc.temperatures[m]:.2f}  {b4:.3f} -> {af:.3f}")
        cols = ["text_only", "audio_only", "face_only"] + MAIN
        for name, tab in (("Accuracy", acc), ("Macro-F1", f1)):
            print(f"\n{name} (test, n={n_test}, seed {seed})")
            print(f"{'scenario':30s}" + "".join(f"{c[:17]:>18s}" for c in cols))
            for s in SCENARIOS:
                print(f"{s:30s}" + "".join(
                    f"{'n/a':>18s}" if np.isnan(tab[(s, c)]) else f"{tab[(s, c)]:18.3f}" for c in cols))
        ac = list(abl)
        print("\nAblation: Macro-F1 when ADDING one factor to calibrated equal weights (linear pooling)")
        print(f"{'scenario':30s}" + "".join(f"{c[:15]:>16s}" for c in ac))
        for s in SCENARIOS:
            print(f"{s:30s}" + "".join(f"{f1[(s, c)]:16.3f}" for c in ac))

        # Does the fused output tell us when it is wrong?
        print("\nError detection on the conflict scenario (AUROC; 0.5 = useless):")
        data = scen["conflict_30pct"]
        f = methods["reliability_linear"]
        b = run_fuser(f, data)
        err = b.pred != data["y"]
        print(f"  1 - fused confidence : {auroc(1 - b.confidence, err):.3f}")
        print(f"  conflict_score       : {auroc(b.conflict_score, err):.3f}")
        print(f"  conflict flag rate   : conflict scenario {b.conflict.mean():.1%}, "
              f"clean {run_fuser(f, scen['clean']).conflict.mean():.1%}  (threshold {f.conflict_threshold} bits)")
        keep = b.confidence >= np.quantile(b.confidence, 0.2)
        print(f"  accuracy overall {accuracy(data['y'], b.pred):.3f} -> {accuracy(data['y'][keep], b.pred[keep]):.3f} "
              f"after abstaining on the 20% least-confident")
    return f1, best


def main(n_dev=3000, n_test=6000, n_seeds=5):
    f1_runs = []
    for seed in range(n_seeds):
        f1, best = run_once(seed, n_dev, n_test, verbose=(seed == 0))
        f1_runs.append(f1)

    print(f"\n\n=== Macro-F1 mean +- sd over {n_seeds} independent dev/test draws ===")
    cols = MAIN
    print(f"{'scenario':30s}" + "".join(f"{c[:17]:>20s}" for c in cols) + f"{'log - equal+calib':>22s}")
    summary = {}
    for s in SCENARIOS:
        line = f"{s:30s}"
        for c in cols:
            v = np.array([r[(s, c)] for r in f1_runs])
            line += f"{v.mean():12.3f} +-{v.std():.3f}"
            summary[(s, c)] = (float(v.mean()), float(v.std()))
        d = np.array([r[(s, "reliability_log")] - r[(s, "equal+calibration")] for r in f1_runs])
        d2 = np.array([r[(s, "reliability_linear")] - r[(s, "equal+calibration")] for r in f1_runs])
        line += f"   {d.mean():+.3f} (wins {int((d > 0).sum())}/{n_seeds})"
        summary[(s, "log_minus_calib")] = (float(d.mean()), float(d.std()), int((d > 0).sum()))
        summary[(s, "linear_minus_calib")] = (float(d2.mean()), float(d2.std()), int((d2 > 0).sum()))
        print(line)

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    with open(out / "synthetic_results.json", "w") as fh:
        json.dump({
            "note": "SIMULATED model outputs -- validates fusion mechanics only, not real-dataset accuracy.",
            "n_dev": n_dev, "n_test": n_test, "n_seeds": n_seeds,
            "macro_f1_mean_sd": {f"{s}|{c}": v for (s, c), v in summary.items()},
        }, fh, indent=2)
    print(f"\nSaved {out / 'synthetic_results.json'}")


if __name__ == "__main__":
    main()
