"""Mechanics of the real-prediction evaluation script, using SYNTHETIC files (never real performance)."""
import csv
import json

import numpy as np
import pytest

from fusion import MELD_LABELS, FusionInputError
from fusion.experiments import evaluate_real as ev

K = 7


def _write_split(d, split, n, seed, mods=("text", "audio"), drop=None, order=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    ids = [f"dia{i // 6}_utt{i % 6}" for i in range(n)]
    with open(d / f"labels_{split}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "label", "dialogue_id"])
        for i, s in enumerate(ids):
            w.writerow([s, MELD_LABELS[y[i]], i // 6])
    for m in mods:
        z = rng.normal(size=(n, K)) * 1.5
        z[np.arange(n), y] += 1.5
        P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        cols = list(order or MELD_LABELS)
        with open(d / f"probs_{split}_{m}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id", *cols])
            for i, s in enumerate(ids):
                if drop and m == drop[0] and i % drop[1] == 0:
                    continue                                   # this model has no output for the sample
                w.writerow([s, *[P[i, MELD_LABELS.index(c)] for c in cols]])
    return ids, y


@pytest.fixture()
def data(tmp_path):
    _write_split(tmp_path, "dev", 600, 1)
    _write_split(tmp_path, "test", 900, 2)
    return tmp_path


def test_end_to_end_runs_and_reports_everything_it_should(data):
    res = ev.evaluate(data)
    assert res["modalities_with_predictions"] == ["text", "audio"] and res["modalities_without_predictions"] == ["face"]
    assert set(res["single_modality_test"]) == {"text", "audio"}
    assert res["candidates"]["selected_on_dev"] in res["candidates"]["test"]
    assert set(res["selected_test"]["modality_subsets_used_at_test"]) == {"text", "audio", "text+audio"}
    r = res["selected_test"]["all_test_rows"]
    assert r["n"] == 900 and 0.3 < r["accuracy"] <= 1 and np.array(r["confusion_matrix"]).sum() == 900
    assert res["selected_test"]["abstention"]["test_coverage"] > 0.5
    assert all(v["sha256"] for v in res["provenance"].values())
    json.dumps(res)                                            # serialisable


def test_fusion_beats_the_weaker_single_modality_on_independent_synthetic_noise(data):
    res = ev.evaluate(data)
    fused = res["selected_test"]["all_test_rows"]["accuracy"]
    assert fused > max(d["accuracy"] for d in res["single_modality_test"].values())


def test_missing_samples_are_unavailable_not_filled_in(tmp_path):
    _write_split(tmp_path, "dev", 600, 1, drop=("audio", 5))
    _write_split(tmp_path, "test", 900, 2, drop=("audio", 4))
    res = ev.evaluate(tmp_path)
    assert res["single_modality_test"]["audio"]["n"] == 900 - len(range(0, 900, 4))
    assert res["single_modality_test"]["text"]["n"] == 900
    assert res["selected_test"]["all_test_rows"]["n"] == 900                  # fused still answers via text
    assert res["selected_test"]["rows_where_every_modality_has_output"]["n"] == res["single_modality_test"]["audio"]["n"]
    assert res["selected_test"]["summary"]["status"]["audio"]["missing"] == len(range(0, 900, 4))


def test_column_order_does_not_matter(tmp_path):
    _write_split(tmp_path, "dev", 600, 1, order=list(reversed(MELD_LABELS)))
    _write_split(tmp_path, "test", 900, 2, order=list(reversed(MELD_LABELS)))
    a = ev.evaluate(tmp_path)
    ref = tmp_path / "ref"
    ref.mkdir()
    _write_split(ref, "dev", 600, 1)
    _write_split(ref, "test", 900, 2)
    b = ev.evaluate(ref)
    assert a["selected_test"]["all_test_rows"]["accuracy"] == pytest.approx(b["selected_test"]["all_test_rows"]["accuracy"])


@pytest.mark.parametrize("break_it, msg", [
    ("no_files", "no prediction files"),
    ("one_split_only", "needs both"),
    ("unknown_id", "not in the labels file"),
    ("duplicate_id", "duplicate"),
    ("missing_column", "missing columns"),
    ("extra_column", "unexpected columns"),
    ("non_numeric", "non-numeric"),
    ("bad_label", "labels not in"),
    ("no_labels", "not found"),
])
def test_bad_inputs_raise_a_clear_error(data, break_it, msg):
    p = data / "probs_dev_text.csv"
    rows = list(csv.reader(open(p)))
    def rewrite(rows_):
        with open(p, "w", newline="") as fh:
            csv.writer(fh).writerows(rows_)
    if break_it == "no_files":
        for f in data.glob("probs_*.csv"):
            f.unlink()
    elif break_it == "one_split_only":
        (data / "probs_test_audio.csv").unlink()
    elif break_it == "unknown_id":
        rows[1][0] = "zzz"; rewrite(rows)
    elif break_it == "duplicate_id":
        rows[2][0] = rows[1][0]; rewrite(rows)
    elif break_it == "missing_column":
        rewrite([r[:-1] for r in rows])
    elif break_it == "extra_column":
        rewrite([r + ["love" if i == 0 else "0.1"] for i, r in enumerate(rows)])
    elif break_it == "non_numeric":
        rows[1][2] = "abc"; rewrite(rows)
    elif break_it == "bad_label":
        lp = data / "labels_dev.csv"
        lr = list(csv.reader(open(lp)))
        lr[1][1] = "love"
        csv.writer(open(lp, "w", newline="")).writerows(lr)
    elif break_it == "no_labels":
        (data / "labels_test.csv").unlink()
    with pytest.raises(FusionInputError, match=msg):
        ev.evaluate(data)


def test_cli_returns_nonzero_and_writes_nothing_on_error(tmp_path, capsys):
    out = tmp_path / "o.json"
    assert ev.main(["--data", str(tmp_path), "--out", str(out)]) == 2
    assert not out.exists() and "ERROR" in capsys.readouterr().err


def test_cli_success_writes_json_and_accepts_a_config(data, tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"model_priors": {"text": "uniform"}, "target_prior": "dev"}))
    out = tmp_path / "res.json"
    assert ev.main(["--data", str(data), "--config", str(cfg), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["base_config"]["target_prior"] == "dev"
    assert "selected on dev" in capsys.readouterr().out


def test_custom_files_and_split_prefixed_ids(tmp_path):
    """A teammate's layout: different filenames, ids like dev_dia0_utt0."""
    d = tmp_path / "labels"
    d.mkdir()
    _write_split(d, "dev", 600, 1, mods=("text",))
    _write_split(d, "test", 900, 2, mods=("text",))
    other = tmp_path / "theirs"
    other.mkdir()
    _write_split(other, "dev", 600, 1, mods=("audio",))
    _write_split(other, "test", 900, 2, mods=("audio",))
    for split in ("dev", "test"):                                    # give their ids the split prefix, rename the file
        rows = list(csv.reader(open(other / f"probs_{split}_audio.csv")))
        rows = [rows[0]] + [[f"{split}_{r[0]}", *r[1:]] for r in rows[1:]]
        csv.writer(open(other / f"speech_{split}.csv", "w", newline="")).writerows(rows)
    spec = {"audio": (other / "speech_dev.csv", other / "speech_test.csv")}
    with pytest.raises(FusionInputError, match="not in the labels file"):          # prefix not stripped -> loud failure
        ev.evaluate(d, probs_files=spec)
    res = ev.evaluate(d, probs_files=spec, strip_split_prefix=True)
    assert res["modalities_with_predictions"] == ["text", "audio"]
    assert res["single_modality_test"]["audio"]["n"] == 900


def test_cli_probs_option_and_its_errors(data, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert ev.main(["--data", str(data), "--probs", "audio", "--out", str(out)]) == 2
    assert "MODALITY=DEV_CSV,TEST_CSV" in capsys.readouterr().err
    assert ev.main(["--data", str(data), "--probs", f"speech={data}/a.csv,{data}/b.csv", "--out", str(out)]) == 2
    assert "unknown modality" in capsys.readouterr().err
    ok = ev.main(["--data", str(data), "--probs", f"audio={data}/probs_dev_audio.csv,{data}/probs_test_audio.csv",
                  "--out", str(out)])
    assert ok == 0


def test_paired_bootstrap_detects_a_real_gain_and_a_null(data):
    res = ev.evaluate(data)
    b = res["selected_test"]["fused_minus_single_modality_paired_bootstrap"]
    assert set(b) == {"text", "audio"}
    for m in b:                                                     # independent synthetic noise: fusion clearly helps
        assert b[m]["accuracy"]["lo95"] > 0 and b[m]["macro_f1"]["lo95"] > 0
    y = np.array([0, 1, 2, 3] * 50)
    g = np.repeat(np.arange(50), 4)
    same = ev.paired_bootstrap(y, g, y.copy(), y.copy(), n_boot=200)
    assert same["accuracy"]["mean"] == 0 and same["accuracy"]["lo95"] == 0 == same["accuracy"]["hi95"]


def test_bootstrap_resamples_whole_dialogues():
    y = np.zeros(8, dtype=int)
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])                         # right on dialogue 0, wrong on dialogue 1
    b = np.zeros(8, dtype=int)
    r = ev.paired_bootstrap(y, g, a, b, n_boot=300)
    assert r["accuracy"]["lo95"] <= -0.5 and r["accuracy"]["hi95"] <= 0   # whole-dialogue resamples only give 0 or -0.5 or -1
