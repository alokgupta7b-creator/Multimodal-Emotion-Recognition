"""Evaluate fusion on REAL model predictions supplied as files. Invents nothing.

    python -m fusion.experiments.evaluate_real --data DIR [--config cfg.json] [--out results/real_eval.json]

Input files in DIR (CSV with a header row; written by the model owners from models trained on ``train`` only):

  labels_dev.csv, labels_test.csv      sample_id,label,dialogue_id          (label = a MELD class name)
  probs_<split>_<modality>.csv         sample_id,anger,disgust,fear,joy,neutral,sadness,surprise
                                       <split> in {dev, test}; <modality> in {text, audio, face}

* ``sample_id`` must be identical across models (e.g. ``dia0_utt0``). A sample missing from a model's file is treated as that
  modality being unavailable for that sample (fusion handles it); it is never filled in. An id that is not in the labels file,
  a duplicate id, a missing or unknown class column, or a non-numeric cell is an error.
* A modality needs BOTH its dev and test file, otherwise it is an error. With no prediction files at all the script stops.
* Class columns are matched by NAME, so column order does not matter. Probabilities are used as given (rows are validated and
  renormalised by the fuser; invalid rows are dropped with a reason, see ``summary`` in the output).
* ``--config`` is a JSON dict for ``FusionConfig`` (e.g. ``{"model_priors": {"text": "uniform"}, "target_prior": "dev"}``).

Protocol: everything is chosen on DEV only (candidate configuration by dialogue-grouped out-of-fold macro-F1; abstention
threshold and conflict-flag decision from the same out-of-fold scores), then ONE frozen choice is reported on TEST.
The script cannot verify that the files really come from the stated models or that those were trained on ``train`` only: it
records each file's SHA-256 and row count in the output, and provenance is the caller's responsibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import (  # noqa: E402
    MELD_LABELS, MODALITIES, FusionConfig, FusionInputError, ReliabilityAwareFuser, accuracy,
    expected_calibration_error, macro_f1, out_of_fold_confidence, select_conflict_threshold,
    select_min_confidence,
)
from fusion.calibration import nll  # noqa: E402
from fusion.metrics import confusion_matrix, per_class_prf  # noqa: E402

K = len(MELD_LABELS)


# ----------------------------------------------------------------------------- loading
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_labels(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    need = {"sample_id", "label", "dialogue_id"}
    if not rows or not need <= set(rows[0]):
        raise FusionInputError(f"{path.name}: needs a header with columns {sorted(need)} and at least one row")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise FusionInputError(f"{path.name}: duplicate sample_id values")
    idx = {l: i for i, l in enumerate(MELD_LABELS)}
    bad = sorted({r["label"] for r in rows if r["label"].strip().lower() not in idx})
    if bad:
        raise FusionInputError(f"{path.name}: labels not in {list(MELD_LABELS)}: {bad[:5]}")
    y = np.array([idx[r["label"].strip().lower()] for r in rows])
    return ids, y, np.array([r["dialogue_id"] for r in rows])


def read_probs(path: Path, ids: List[str], strip_prefix: Optional[str] = None) -> np.ndarray:
    """(N, K) in MELD_LABELS order, aligned to ``ids``; NaN rows for ids the file does not contain.

    ``strip_prefix`` (e.g. ``"dev_"``) is removed from the start of each file sample_id before matching, for files whose ids
    carry a split prefix (``dev_dia0_utt0``). An id that does not start with it is left as it is (and so fails the match)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        missing = [c for c in ("sample_id", *MELD_LABELS) if c not in cols]
        if missing:
            raise FusionInputError(f"{path.name}: missing columns {missing}; expected sample_id + {list(MELD_LABELS)}")
        extra = [c for c in cols if c not in ("sample_id", *MELD_LABELS)]
        if extra:
            raise FusionInputError(f"{path.name}: unexpected columns {extra} (a class this label set does not have?)")
        pos = {s: i for i, s in enumerate(ids)}
        out = np.full((len(ids), K), np.nan)
        seen = set()
        for ln, r in enumerate(reader, start=2):
            sid = r["sample_id"]
            if strip_prefix and sid.startswith(strip_prefix):
                sid = sid[len(strip_prefix):]
            if sid not in pos:
                raise FusionInputError(f"{path.name} line {ln}: sample_id {sid!r} is not in the labels file")
            if sid in seen:
                raise FusionInputError(f"{path.name} line {ln}: duplicate sample_id {sid!r}")
            seen.add(sid)
            try:
                out[pos[sid]] = [float(r[c]) if r[c].strip() != "" else np.nan for c in MELD_LABELS]
            except ValueError:
                raise FusionInputError(f"{path.name} line {ln}: non-numeric probability") from None
    return out


# ----------------------------------------------------------------------------- metrics
def prf_macro(y, pred):
    p = per_class_prf(y, pred, K)
    present = p["support"] > 0
    return float(p["precision"][present].mean()), float(p["recall"][present].mean())


def report(y, P):
    """Metrics over rows with a prediction. ``n`` = rows scored."""
    ok = np.isfinite(P).any(axis=1)
    if not ok.any():
        return {"n": 0}
    y, P = y[ok], P[ok]
    pred = np.nanargmax(np.where(np.isnan(P), -np.inf, P), axis=1)
    pr, rc = prf_macro(y, pred)
    full = np.isfinite(P).all(axis=1)
    out = {"n": int(ok.sum()), "accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, K),
           "macro_precision": pr, "macro_recall": rc,
           "confusion_matrix": confusion_matrix(y, pred, K).tolist()}
    if full.any():
        out["nll"] = nll(P[full], y[full])
        out["ece"] = expected_calibration_error(P[full], y[full])
    return out


WEIGHTINGS = {"": {}, "+priors": {"use_priors": True}, "+confidence": {"confidence_power": 1.0},
              "+agreement": {"consistency_power": 1.0},
              "+all": {"use_priors": True, "confidence_power": 1.0, "consistency_power": 1.0}}


def candidates(base: FusionConfig) -> Dict[str, FusionConfig]:
    """pooling x reliability weighting. Quality weighting is not a candidate: it needs quality scores nobody supplies yet.
    ``calibrate_fused`` is not a candidate either (it never changes the argmax, so it cannot change the dev ranking); it is
    reported separately for the selected configuration."""
    out = {}
    for pooling, (wn, kw) in itertools.product(("linear", "log"), WEIGHTINGS.items()):
        out[f"{pooling}{wn}"] = replace(base, pooling=pooling, calibrate_fused=False, **kw)
    return out


def paired_bootstrap(y, g, pred_a, pred_b, n_boot=1000, seed=0):
    """Fused-minus-single differences in accuracy and macro-F1 with 95 % intervals, resampling whole DIALOGUES
    (utterances of one dialogue are not independent). Rows without a prediction (-1) count as wrong for that side."""
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    rows_of = {k: np.flatnonzero(g == k) for k in groups}
    da, df = [], []
    for _ in range(n_boot):
        r = np.concatenate([rows_of[k] for k in rng.choice(groups, len(groups))])
        da.append(accuracy(y[r], pred_a[r]) - accuracy(y[r], pred_b[r]))
        df.append(macro_f1(y[r], pred_a[r], K) - macro_f1(y[r], pred_b[r], K))
    ci = lambda a: {"mean": float(np.mean(a)), "lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5))}
    return {"accuracy": ci(da), "macro_f1": ci(df), "n_boot": n_boot}


def fit_fuse(cfg, dev, y_dev, test, subset=None):
    f = ReliabilityAwareFuser(cfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f.fit(dev, y_dev)
        r = f.fuse_batch({m: (test[m] if subset is None or m in subset else None) for m in test})
    return f, r, sorted({str(x.message) for x in w})


# ----------------------------------------------------------------------------- main
def evaluate(data_dir: Path, config: Optional[dict] = None, seed: int = 0,
             probs_files: Optional[Dict[str, Tuple[Path, Path]]] = None, strip_split_prefix: bool = False) -> dict:
    """``probs_files`` maps a modality to ``(dev_csv, test_csv)`` paths that replace the default
    ``probs_<split>_<modality>.csv`` in ``data_dir``; ``strip_split_prefix`` removes ``dev_`` / ``test_`` from file sample ids."""
    data_dir = Path(data_dir)
    probs_files = {m: (Path(a), Path(b)) for m, (a, b) in (probs_files or {}).items()}
    unknown = sorted(set(probs_files) - set(MODALITIES))
    if unknown:
        raise FusionInputError(f"unknown modality {unknown}; expected one of {list(MODALITIES)}")
    files, prov = {}, {}
    for split in ("dev", "test"):
        p = data_dir / f"labels_{split}.csv"
        if not p.exists():
            raise FusionInputError(f"{p} not found")
        files[split] = read_labels(p)
        prov[p.name] = {"sha256": _sha(p), "rows": len(files[split][0])}
    dev_ids, y_dev, g_dev = files["dev"]
    te_ids, y_te, g_te = files["test"]
    mods = []
    dev_p: Dict[str, np.ndarray] = {}
    te_p: Dict[str, np.ndarray] = {}
    for m in MODALITIES:
        pd_, pt_ = probs_files.get(m, (data_dir / f"probs_dev_{m}.csv", data_dir / f"probs_test_{m}.csv"))
        if pd_.exists() != pt_.exists():
            raise FusionInputError(f"modality {m!r}: needs both {pd_.name} and {pt_.name} (found only one)")
        if not pd_.exists():
            continue
        mods.append(m)
        dev_p[m] = read_probs(pd_, dev_ids, "dev_" if strip_split_prefix else None)
        te_p[m] = read_probs(pt_, te_ids, "test_" if strip_split_prefix else None)
        for p, arr in ((pd_, dev_p[m]), (pt_, te_p[m])):
            prov[p.name] = {"sha256": _sha(p), "rows_with_predictions": int(np.isfinite(arr).any(axis=1).sum())}
    if not mods:
        raise FusionInputError(f"no prediction files (probs_<split>_<modality>.csv) in {data_dir}: nothing to evaluate")

    base = FusionConfig.from_dict(config) if config else FusionConfig()
    n_groups = len(np.unique(g_dev))
    folds = min(5, n_groups)
    if folds < 2:
        raise FusionInputError("need at least 2 dialogues in the dev labels for grouped out-of-fold selection")

    res: dict = {"provenance": prov, "modalities_with_predictions": mods,
                 "modalities_without_predictions": [m for m in MODALITIES if m not in mods],
                 "n_dev": len(dev_ids), "n_test": len(te_ids), "n_dev_dialogues": n_groups,
                 "base_config": base.to_dict()}

    # 1. single modalities (test rows where the model produced an output)
    res["single_modality_test"] = {m: report(y_te, te_p[m]) for m in mods}
    res["single_modality_dev"] = {m: report(y_dev, dev_p[m]) for m in mods}

    # 2. candidates: out-of-fold dev macro-F1 selects, test is reported for all
    cands = candidates(base)
    oof_scores, test_rows = {}, {}
    for name, cfg in cands.items():
        oof = out_of_fold_confidence(cfg, dev_p, y_dev, g_dev, n_folds=folds, seed=seed)
        ok = oof["pred"] >= 0
        oof_scores[name] = float(macro_f1(y_dev[ok], oof["pred"][ok], K)) if ok.any() else float("nan")
        _, r, warns = fit_fuse(cfg, dev_p, y_dev, te_p)
        test_rows[name] = {**report(y_te, r.probs), "fit_warnings": warns}
    sel = max(oof_scores, key=lambda k: (-1 if np.isnan(oof_scores[k]) else oof_scores[k]))
    eq = ReliabilityAwareFuser(base.equal_weight_baseline())
    eq_r = eq.fuse_batch({m: te_p[m] for m in mods})
    res["equal_weight_baseline_test"] = report(y_te, eq_r.probs)
    res["candidates"] = {"dev_out_of_fold_macro_f1": oof_scores, "test": test_rows, "selected_on_dev": sel}

    # 3. the selected configuration: availability patterns, abstention, conflict
    cfg = cands[sel]
    f, full, _ = fit_fuse(cfg, dev_p, y_dev, te_p)
    _, full_ft, _ = fit_fuse(replace(cfg, calibrate_fused=True), dev_p, y_dev, te_p)
    both = np.all([np.isfinite(te_p[m]).any(axis=1) for m in mods], axis=0)
    res["selected_test"] = {
        "all_test_rows": report(y_te, full.probs),
        "rows_where_every_modality_has_output": {"n": int(both.sum()), **({} if not both.any() else {
            "fused": report(y_te[both], full.probs[both]),
            "equal_weight": report(y_te[both], eq_r.probs[both]),
            **{m: report(y_te[both], te_p[m][both]) for m in mods}})},
        "summary": full.summary(),
        "with_fused_temperature": report(y_te, full_ft.probs),      # same predictions, probabilities re-calibrated on dev
    }
    per_subset = {}
    for r_ in range(1, len(mods) + 1):
        for sub in itertools.combinations(mods, r_):
            _, rr, _ = fit_fuse(cfg, dev_p, y_dev, te_p, subset=set(sub))
            per_subset["+".join(sub)] = report(y_te, rr.probs)
    res["selected_test"]["modality_subsets_used_at_test"] = per_subset

    if len(mods) > 1:                       # does fusing beat each model alone, under the SAME configuration?
        pf = full.pred
        boot = {}
        for m in mods:
            _, rm, _ = fit_fuse(cfg, dev_p, y_dev, te_p, subset={m})
            boot[m] = paired_bootstrap(y_te, g_te, pf, rm.pred, seed=seed)
        res["selected_test"]["fused_minus_single_modality_paired_bootstrap"] = boot
        if len(mods) > 2:                   # what does each modality add to the others?
            loo = {}
            for m in mods:
                _, rm, _ = fit_fuse(cfg, dev_p, y_dev, te_p, subset=set(mods) - {m})
                loo[f"without_{m}"] = paired_bootstrap(y_te, g_te, pf, rm.pred, seed=seed)
            res["selected_test"]["fused_minus_leave_one_modality_out_paired_bootstrap"] = loo
        res["selected_test"]["fused_minus_equal_weight_baseline_paired_bootstrap"] = paired_bootstrap(
            y_te, g_te, pf, eq_r.pred, seed=seed)
    oof = out_of_fold_confidence(cfg, dev_p, y_dev, g_dev, n_folds=folds, seed=seed)
    usable = oof["pred"] >= 0
    ab = {}
    if usable.sum() >= 30:
        thr, info = select_min_confidence(oof["confidence"][usable], oof["correct"][usable], coverage=0.8)
        keep = (full.confidence >= thr) & (full.pred >= 0)
        okt = full.pred == y_te
        ab = {"coverage_target": 0.8, "threshold_from_dev_oof": thr, "dev_oof": info,
              "test_coverage": float(keep.mean()), "test_error_among_kept": float((~okt[keep]).mean()) if keep.any() else None,
              "test_error_all": float((~okt[full.pred >= 0]).mean())}
    res["selected_test"]["abstention"] = ab
    thr_c, info_c = select_conflict_threshold(oof["conflict_score"], oof["correct"], oof["n_available"])
    res["selected_test"]["conflict_rule"] = {"enable_with_threshold": thr_c, "dev_oof": info_c}
    res["notes"] = ("REAL predictions from the listed files. Provenance (that they come from the named models, trained on "
                    "train only, on identical sample ids) is NOT verified here.")
    return res


def _print(res: dict) -> None:
    f = lambda d, k: f"{d[k]:.3f}" if k in d else "  n/a"
    print(f"REAL predictions | modalities: {res['modalities_with_predictions']} | without predictions: "
          f"{res['modalities_without_predictions']} | dev {res['n_dev']} ({res['n_dev_dialogues']} dialogues), test {res['n_test']}")
    print(f"\n{'TEST':34s} {'n':>6s} {'acc':>7s} {'macroF1':>8s} {'macroP':>7s} {'macroR':>7s} {'NLL':>6s} {'ECE':>6s}")
    row = lambda name, d: print(f"{name:34s} {d.get('n', 0):6d} {f(d,'accuracy'):>7s} {f(d,'macro_f1'):>8s} "
                                f"{f(d,'macro_precision'):>7s} {f(d,'macro_recall'):>7s} {f(d,'nll'):>6s} {f(d,'ece'):>6s}")
    for m, d in res["single_modality_test"].items():
        row(f"{m} alone (rows it covers)", d)
    row("equal-weight fusion", res["equal_weight_baseline_test"])
    sel = res["candidates"]["selected_on_dev"]
    for n, d in res["candidates"]["test"].items():
        row(f"fusion {n}{' <- selected on dev' if n == sel else ''}", d)
    print("\nselected configuration, by modalities used at test time:")
    for k, d in res["selected_test"]["modality_subsets_used_at_test"].items():
        row(f"  {k}", d)
    for m, b in res["selected_test"].get("fused_minus_single_modality_paired_bootstrap", {}).items():
        a_, f_ = b["accuracy"], b["macro_f1"]
        print(f"fused minus {m} alone (same config, dialogue bootstrap): accuracy {a_['mean']:+.3f} [{a_['lo95']:+.3f}, {a_['hi95']:+.3f}], "
              f"macro-F1 {f_['mean']:+.3f} [{f_['lo95']:+.3f}, {f_['hi95']:+.3f}]")
    show = lambda b: (f"accuracy {b['accuracy']['mean']:+.3f} [{b['accuracy']['lo95']:+.3f}, {b['accuracy']['hi95']:+.3f}], "
                      f"macro-F1 {b['macro_f1']['mean']:+.3f} [{b['macro_f1']['lo95']:+.3f}, {b['macro_f1']['hi95']:+.3f}]")
    for k, b in res["selected_test"].get("fused_minus_leave_one_modality_out_paired_bootstrap", {}).items():
        print(f"fused minus fused {k.replace('_', ' ')}: {show(b)}")
    if "fused_minus_equal_weight_baseline_paired_bootstrap" in res["selected_test"]:
        print(f"fused minus equal-weight baseline: {show(res['selected_test']['fused_minus_equal_weight_baseline_paired_bootstrap'])}")
    ft = res["selected_test"]["with_fused_temperature"]
    print(f"probabilities with calibrate_fused=True (same predictions): NLL {ft.get('nll', float('nan')):.3f}, ECE {ft.get('ece', float('nan')):.3f}")
    ab = res["selected_test"]["abstention"]
    if ab:
        print(f"\nabstention (threshold {ab['threshold_from_dev_oof']:.3f} from dev out-of-fold, target coverage 0.8): test coverage "
              f"{ab['test_coverage']:.3f}, error among kept {ab['test_error_among_kept']:.3f} vs {ab['test_error_all']:.3f} overall")
    c = res["selected_test"]["conflict_rule"]
    print("conflict flag:", "enable at threshold %.3f" % c["enable_with_threshold"] if c["enable_with_threshold"] is not None
          else f"leave off ({c['dev_oof'].get('reason', 'not informative')})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--config", type=Path, help="JSON dict of FusionConfig options")
    ap.add_argument("--probs", action="append", default=[], metavar="MODALITY=DEV_CSV,TEST_CSV",
                    help="use these files for a modality instead of probs_<split>_<modality>.csv in --data (repeatable)")
    ap.add_argument("--strip-split-prefix", action="store_true",
                    help="drop a leading 'dev_' / 'test_' from sample ids in the probability files")
    ap.add_argument("--out", type=Path, default=Path("fusion/results/real_eval.json"))
    a = ap.parse_args(argv)
    try:
        pf = {}
        for spec in a.probs:
            if "=" not in spec or "," not in spec.split("=", 1)[1]:
                raise FusionInputError(f"--probs {spec!r}: expected MODALITY=DEV_CSV,TEST_CSV")
            m, paths = spec.split("=", 1)
            d, t = paths.split(",", 1)
            pf[m] = (Path(d), Path(t))
        res = evaluate(a.data, json.loads(a.config.read_text()) if a.config else None, probs_files=pf,
                       strip_split_prefix=a.strip_split_prefix)
    except (FusionInputError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    _print(res)
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
