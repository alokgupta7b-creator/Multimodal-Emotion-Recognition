"""Does the dev-derived conflict rule switch on only when disagreement is informative? SIMULATED, not real data.

For each setting (modality simulator from ``correlated_errors``) fuse dev data with dialogue-grouped out-of-fold
fitting and apply ``select_conflict_threshold``. Also reports the fixed default rule (conflict_threshold=0.25)
for comparison: the share of rows it flags and the error rate among flagged vs unflagged rows.

    python -m fusion.experiments.conflict_rule     # ~15 s, writes fusion/results/conflict_rule.json
"""
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from fusion import FusionConfig, out_of_fold_confidence, select_conflict_threshold  # noqa: E402
from fusion.experiments.correlated_errors import MODS, simulate  # noqa: E402

N, SEEDS = 4000, range(5)
SETTINGS = {
    "independent errors": dict(rho=0.0, p_conf=0.0, conf_strength=0.0),
    "shared noise rho=0.6": dict(rho=0.6, p_conf=0.0, conf_strength=0.0),
    "one modality misleading on 25% of rows": dict(rho=0.0, p_conf=0.0, conf_strength=0.0, mislead=0.25),
    "shared confuser 30%": dict(rho=0.3, p_conf=0.3, conf_strength=3.0),
}


def data(seed, kw):
    kw = dict(kw)
    mislead = kw.pop("mislead", 0.0)
    y, P = simulate(N, seed, **kw)
    if mislead:                                       # audio confidently names a wrong class on a random subset
        rng = np.random.default_rng(seed + 999)
        rows = rng.uniform(size=N) < mislead
        wrong = (y + rng.integers(1, 7, N)) % 7
        bad = np.full((N, 7), 0.02); bad[np.arange(N), wrong] = 0.88
        P["audio"] = np.where(rows[:, None], bad / bad.sum(1, keepdims=True), P["audio"])
    return y, P


def main():
    out, groups = {}, np.repeat(np.arange(N // 8), 8)
    print("SIMULATED. Dev-derived conflict rule vs the fixed default (threshold 0.25); 5 seeds")
    print(f"{'setting':42s} {'rule accepts':>12s} {'lift':>6s} | default flags: share  err flagged / unflagged")
    for name, kw in SETTINGS.items():
        acc, lifts, shares, e1, e0 = 0, [], [], [], []
        for seed in SEEDS:
            y, P = data(300 + seed, kw)
            oof = out_of_fold_confidence(FusionConfig(pooling="linear"), P, y, groups, 5, seed)
            thr, info = select_conflict_threshold(oof["conflict_score"], oof["correct"], oof["n_available"])
            acc += thr is not None
            lifts.append(info.get("lift", np.nan))
            fl = oof["conflict_score"] >= 0.25
            shares.append(fl.mean())
            e1.append((~oof["correct"][fl]).mean() if fl.any() else np.nan)
            e0.append((~oof["correct"][~fl]).mean())
        out[name] = dict(accepted_of_5=int(acc), mean_lift=float(np.nanmean(lifts)), default_flag_share=float(np.mean(shares)),
                         default_err_flagged=float(np.nanmean(e1)), default_err_unflagged=float(np.mean(e0)))
        print(f"{name:42s} {acc:>9d}/5 {np.nanmean(lifts):6.2f} | {np.mean(shares):17.2f}  {np.nanmean(e1):.2f} / {np.mean(e0):.2f}")
    (Path(__file__).resolve().parents[1] / "results" / "conflict_rule.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
