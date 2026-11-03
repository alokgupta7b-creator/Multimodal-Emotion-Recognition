"""Isolated effect of each fusion option on SIMULATED data.

Same simulator and protocol as ``synthetic_experiment.py`` (5 independent dev/test draws); every
configuration differs from its control in exactly ONE option, so each effect is attributable.
Nothing is tuned here: temperatures/priors are fitted on the dev draw, scores are on the test draw.
The simulation assumes independent errors between modalities, which favours log pooling by
construction. NOT evidence about real data.

    python -m fusion.experiments.isolated_options     # ~15 s, writes fusion/results/isolated_options.json
"""
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from fusion import macro_f1  # noqa: E402
from fusion.experiments import synthetic_experiment as ex  # noqa: E402

K, SC, SEEDS = ex.K, ex.SCENARIOS, range(5)
CONTROL_LIN = dict()                                  # library defaults: calibrated equal weights, linear
CONTROL_LOG = dict(pooling="log")
CASES = {  # name -> (cfg, control name)
    "linear (control)": (CONTROL_LIN, None),
    "log pooling": (CONTROL_LOG, "linear (control)"),
    "linear +priors": (dict(use_priors=True), "linear (control)"),
    "linear +confidence": (dict(confidence_power=1.0), "linear (control)"),
    "linear +quality": (dict(quality_power=1.0), "linear (control)"),
    "linear +consistency": (dict(consistency_power=1.0), "linear (control)"),
    "log +priors": (dict(pooling="log", use_priors=True), "log pooling"),
    "log +confidence": (dict(pooling="log", confidence_power=1.0), "log pooling"),
    "log +quality": (dict(pooling="log", quality_power=1.0), "log pooling"),
    "log +consistency": (dict(pooling="log", consistency_power=1.0), "log pooling"),
    "no calibration (linear)": (dict(use_calibration=False), "linear (control)"),
}
res = {n: {s: [] for s in SC} for n in CASES}
for seed in SEEDS:
    dev, test = ex.simulate(3000, 10 * seed + 1), ex.simulate(6000, 10 * seed + 2)
    data = {s: ex.make_scenario(test, s, 1000 * seed + i) for i, s in enumerate(SC)}
    for name, (cfg, _) in CASES.items():
        f = ex.make_fuser(cfg, dev)
        for s in SC:
            res[name][s].append(macro_f1(data[s]["y"], ex.run_fuser(f, data[s]).pred, K))

print("Macro-F1 mean over 5 draws (SIMULATED).  Each row after the control shows the DIFFERENCE to its control, and wins/5.\n")
print(f"{'scenario':30s}" + "".join(f"{n[:19]:>21s}" for n in CASES))
for s in SC:
    line = f"{s:30s}"
    for n, (cfg, ctrl) in CASES.items():
        m = np.mean(res[n][s])
        if ctrl is None:
            line += f"{m:21.3f}"
        else:
            d = np.array(res[n][s]) - np.array(res[ctrl][s])
            line += f"{d.mean():+12.3f} ({(d > 1e-9).sum()}/5)"
    print(line)
print("\nMean difference over the 9 scenarios / scenarios where the option wins on average / loses on average:")
for n, (cfg, ctrl) in CASES.items():
    if ctrl is None: continue
    dm = np.array([np.mean(np.array(res[n][s]) - np.array(res[ctrl][s])) for s in SC])
    print(f"  {n:26s} vs {ctrl:17s}: mean {dm.mean():+.4f}   wins {(dm > 0.002).sum()}  ties {(abs(dm) <= 0.002).sum()}  losses {(dm < -0.002).sum()}   range {dm.min():+.3f} .. {dm.max():+.3f}")

out = Path(__file__).resolve().parents[1] / "results" / "isolated_options.json"
out.parent.mkdir(exist_ok=True)
json.dump({
    "note": "SIMULATED model outputs (independent errors by construction); not real-dataset performance.",
    "n_draws": len(list(SEEDS)), "controls": {n: c for n, (_, c) in CASES.items()},
    "configs": {n: cfg for n, (cfg, _) in CASES.items()},
    "macro_f1_mean_per_draw": {n: {s: [round(float(v), 4) for v in res[n][s]] for s in SC} for n in CASES},
}, open(out, "w"), indent=2)
print(f"\nSaved {out}")
