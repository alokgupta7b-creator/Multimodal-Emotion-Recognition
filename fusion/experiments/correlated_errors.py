"""Pooling under CORRELATED modality errors, on SIMULATED data (not real-data performance).

The earlier simulator makes modality errors independent, which favours log pooling (a product of experts
assumes conditional independence). Real modalities share errors: the same ambiguous line of dialogue misleads
the text model and the tone of voice together, and a "confuser" emotion can fool several at once. Here:

  logits_m = strength_m * q_m * onehot(y)  +  sqrt(1-rho) * eps_m  +  sqrt(rho) * eps_shared  +  c * onehot(confuser)

``rho`` is the share of the noise that is common to all modalities; with probability ``p_conf`` a sample has a
shared confuser class (c > 0 pushes every modality towards the same wrong class). Calibration is fitted on a
dev draw from the SAME process; scores are on a separate test draw. Macro-F1 / accuracy / NLL of the fused output.

Pooling candidates (all per-model calibrated, equal weights, uniform priors):
  linear         weighted mean                                    (library default)
  log            product of experts, weights mean 1               (``pooling="log"``)
  log+fusedT     log, then one temperature fitted on the fused dev output (``calibrate_fused=True``)
  linear+fusedT  the same for linear pooling
Argmax (so accuracy / macro-F1) is identical with and without fusedT; it changes NLL, i.e. confidence.

    python -m fusion.experiments.correlated_errors     # ~20 s, writes fusion/results/correlated_errors.json
"""
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from fusion import FusionConfig, MELD_LABELS, ReliabilityAwareFuser, accuracy, macro_f1  # noqa: E402
from fusion.calibration import nll  # noqa: E402

K, N_DEV, N_TEST, SEEDS = len(MELD_LABELS), 4000, 8000, range(5)
MODS = ("text", "audio", "face")
STRENGTH = {"text": 2.4, "audio": 1.9, "face": 1.6}
OVERCONF = {"text": 1.3, "audio": 2.2, "face": 1.8}
PRIOR = np.array([0.11, 0.03, 0.03, 0.17, 0.47, 0.07, 0.12])
PRIOR /= PRIOR.sum()


def softmax(L):
    L = L - L.max(-1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(-1, keepdims=True)


def simulate(n, seed, rho, p_conf, conf_strength, degrade=None):
    """``degrade``: optional (modality, factor): that modality's signal is multiplied by factor (varied degradation)."""
    rng = np.random.default_rng(seed)
    # Balanced classes: no model here embeds a class prior, so uniform labels keep "uniform prior" honest.
    y = rng.integers(0, K, n)
    shared = rng.normal(size=(n, K))
    has_conf = rng.uniform(size=n) < p_conf
    confuser = (y + rng.integers(1, K, n)) % K                  # a class that is not the true one
    probs = {}
    for m in MODS:
        q = rng.beta(5, 2, n)
        s = STRENGTH[m] * (degrade[1] if degrade and degrade[0] == m else 1.0)
        L = (s * q[:, None] * np.eye(K)[y] + np.sqrt(1 - rho) * rng.normal(size=(n, K)) + np.sqrt(rho) * shared)
        L = L + conf_strength * has_conf[:, None] * np.eye(K)[confuser]
        probs[m] = softmax(OVERCONF[m] * L)
    return y, probs


def score(Pf, y):
    pred = Pf.argmax(1)
    return dict(macro_f1=macro_f1(y, pred, K), acc=accuracy(y, pred), nll=nll(Pf, y))


def run(rho, p_conf, conf_strength, degrade=None, subset=MODS):
    agg = {}
    for seed in SEEDS:
        yd, Pd = simulate(N_DEV, 100 + seed, rho, p_conf, conf_strength)            # dev: undegraded, as in practice
        yt, Pt = simulate(N_TEST, 200 + seed, rho, p_conf, conf_strength, degrade)
        res = {}
        eq = ReliabilityAwareFuser(FusionConfig().equal_weight_baseline())
        res["equal weight, uncalibrated"] = score(eq.fuse_batch({m: Pt[m] for m in subset}).probs, yt)
        fits = {}
        for name, pooling, fused_t in (("linear", "linear", False), ("log", "log", False),
                                       ("linear+fusedT", "linear", True), ("log+fusedT", "log", True)):
            f = ReliabilityAwareFuser(FusionConfig(pooling=pooling, calibrate_fused=fused_t)).fit(Pd, yd)
            fits[name] = f
            res[name] = score(f.fuse_batch({m: Pt[m] for m in subset}).probs, yt)
        for k, v in res.items():
            for met, val in v.items():
                agg.setdefault(k, {}).setdefault(met, []).append(val)
    return {k: {met: round(float(np.mean(v)), 4) for met, v in d.items()} for k, d in agg.items()}


def main():
    out = {"_note": ("SIMULATED. Balanced classes, calibration fitted on a dev draw of the same process, mean over "
                     f"{len(list(SEEDS))} seeds. Not real-data performance.")}
    COLS = ("linear", "log", "linear+fusedT", "log+fusedT")
    fmt = lambda r: " ".join(f"{r[k]['macro_f1']:.3f}/{r[k]['nll']:.2f}".rjust(15) for k in COLS)
    print("SIMULATED (not real-data performance). macro-F1 / NLL of the fused output, all three modalities")
    print(f"{'setting':40s} " + " ".join(f"{k:>15s}" for k in COLS))
    settings = {
        "independent errors (rho=0)": dict(rho=0.0, p_conf=0.0, conf_strength=0.0),
        "shared noise rho=0.3": dict(rho=0.3, p_conf=0.0, conf_strength=0.0),
        "shared noise rho=0.6": dict(rho=0.6, p_conf=0.0, conf_strength=0.0),
        "shared noise rho=0.9": dict(rho=0.9, p_conf=0.0, conf_strength=0.0),
        "shared confuser, 20% of samples": dict(rho=0.3, p_conf=0.2, conf_strength=3.0),
        "shared confuser, 40% of samples": dict(rho=0.3, p_conf=0.4, conf_strength=3.0),
    }
    for name, kw in settings.items():
        r = run(**kw)
        out[name] = {"all modalities": r}
        print(f"{name:40s} " + fmt(r))
    print("\nVaried degradation at test time (rho=0.6): one modality's signal multiplied by a factor, calibration fitted on clean dev")
    for mod in ("audio", "face"):
        for fac in (1.0, 0.5, 0.2, 0.0):
            r = run(rho=0.6, p_conf=0.0, conf_strength=0.0, degrade=(mod, fac))
            out.setdefault("degradation rho=0.6", {})[f"{mod} x{fac}"] = r
            print(f"{(mod + ' signal x' + str(fac)):40s} " + fmt(r))
    p = Path(__file__).resolve().parents[1] / "results" / "correlated_errors.json"
    p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
