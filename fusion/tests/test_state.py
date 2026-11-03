"""Saving and loading a fitted fuser (so integration need not refit from dev predictions). Synthetic inputs."""
import json
import warnings

import numpy as np
import pytest

from fusion import FusionConfig, FusionInputError, ReliabilityAwareFuser

K = 7
MODS = ("text", "audio", "face")


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    P = {}
    for m in MODS:
        z = rng.normal(size=(n, K)) * 1.6
        z[np.arange(n), y] += 1.4
        e = np.exp(z - z.max(1, keepdims=True))
        P[m] = e / e.sum(1, keepdims=True)
    return P, y


CONFIGS = [
    FusionConfig(),
    FusionConfig(pooling="log", calibrate_fused=True, use_priors=True, confidence_power=1.0),
    FusionConfig(pooling="log", model_priors={"text": [3, 1, 1, 2, 9, 1, 1]}, target_prior="dev", target_prior_power=0.5),
    FusionConfig(modality_support={"text": ["anger", "fear", "joy", "sadness", "surprise"]}, pooling="log"),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_round_trip_gives_identical_outputs(cfg, tmp_path):
    P, y = _data()
    if "text" in cfg.modality_support:
        mask = np.array([l in cfg.modality_support["text"] for l in cfg.labels])
        P["text"] = np.where(mask, P["text"], np.nan)
        P["text"][:, mask] /= P["text"][:, mask].sum(1, keepdims=True)
    f = ReliabilityAwareFuser(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.fit(P, y)
    path = tmp_path / "fuser.json"
    f.save(path)
    g = ReliabilityAwareFuser.load(path)
    assert g.config == f.config and g.fitted and g.temperatures == f.temperatures and g.priors == f.priors
    assert g.fused_temperature == f.fused_temperature
    a, b = f.fuse_batch(P), g.fuse_batch(P)
    np.testing.assert_array_equal(a.probs, b.probs)
    np.testing.assert_array_equal(a.pred, b.pred)
    np.testing.assert_array_equal(a.uncertain, b.uncertain)


def test_state_is_plain_json_and_does_not_hold_dev_predictions(tmp_path):
    P, y = _data()
    f = ReliabilityAwareFuser(FusionConfig(pooling="log")).fit(P, y)
    text = json.dumps(f.state_dict())
    assert len(text) < 20000                                   # no per-row arrays
    assert json.loads(text)["version"] == 1


def test_unfitted_round_trip_stays_unfitted():
    g = ReliabilityAwareFuser.from_state(ReliabilityAwareFuser(FusionConfig()).state_dict())
    assert not g.fitted and g.temperatures == {m: 1.0 for m in MODS}


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s.pop("temperatures"), "missing keys"),
    (lambda s: s.update(version=99), "version"),
    (lambda s: s["temperatures"].update(text=float("nan")), "finite"),
    (lambda s: s["temperatures"].update(text=-1.0), "finite"),
    (lambda s: s["temperatures"].pop("face"), "exactly the modalities"),
    (lambda s: s["priors"].update(audio=0.0), "> 0"),
    (lambda s: s.update(fused_temperature="hot"), "fused_temperature"),
    (lambda s: s.update(dev_prior=[0.5, 0.5]), "needs 7"),
    (lambda s: s.update(dev_prior=[0.1] * 7), "sum to 1"),
])
def test_malformed_state_is_rejected(mutate, msg):
    P, y = _data()
    s = ReliabilityAwareFuser(FusionConfig(target_prior="dev")).fit(P, y).state_dict()
    s = json.loads(json.dumps(s))
    mutate(s)
    with pytest.raises(FusionInputError, match=msg):
        ReliabilityAwareFuser.from_state(s)


def test_non_dict_and_bad_file_are_rejected(tmp_path):
    with pytest.raises(FusionInputError):
        ReliabilityAwareFuser.from_state([1, 2])
    bad = tmp_path / "x.json"
    bad.write_text("{not json")
    with pytest.raises(FusionInputError, match="not valid JSON"):
        ReliabilityAwareFuser.load(bad)
