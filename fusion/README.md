# Robust Feature Fusion (`fusion/`)

Decision-level fusion of the text, speech and face emotion models.
Owner: Alok. Integration contract: see "Interface" below.

## Interface

Every model must give a probability vector over the **7 MELD classes in alphabetical
order**: `anger, disgust, fear, joy, neutral, sadness, surprise` (`fusion.MELD_LABELS`;
identical to `speech_emotion.model.label_mapping.MELD_CLASS_NAMES`).

```python
from fusion import ReliabilityAwareFuser, ModalityInput, equal_weight_fusion

fuser = ReliabilityAwareFuser(pooling="log", consistency_power=1.0)
fuser.fit({"text": P_text_dev, "audio": P_audio_dev, "face": P_face_dev}, y_dev)  # DEV split only

result = fuser.fuse({
    "text":  text_probs,                                   # (7,) array/list
    "audio": ModalityInput(ser_pred.probability_array, quality=0.8),  # optional quality in [0,1]
    "face":  None,                                         # unavailable -> skipped, never "neutral"
})
result.label, result.confidence, result.available, result.missing,
result.weights, result.conflict, result.uncertain, result.warnings
```

* `fuser.fuse_batch(probs, quality)` does the same for `(N, 7)` matrices (all-NaN row = missing).
* `equal_weight_fusion(...)` is the reference baseline.
* Adapters: `from_ser_prediction(SERPrediction)`, `from_text_probs(probs)`.
* Bad model outputs (NaN, negative, zero-sum, unnormalised) never crash fusion: the modality is
  dropped (or renormalised) and the reason is in `result.warnings`. Wrong vector length or an
  unknown modality name raises `ValueError` (an interface bug that should be fixed, not hidden).

## Method

Per sample, weight of modality *m*: `w_m = prior_m · conf_m^γ · quality_m^β · consistency_m^λ`,
renormalised over the **available** modalities.

| factor | meaning | fitted/learned from |
|---|---|---|
| calibration | temperature scaling of each model's probabilities | dev NLL |
| `prior_m` | dev macro-F1 of the modality | dev |
| `conf_m` | 1 − normalised entropy of the calibrated prediction | per sample |
| `quality_m` | external signal in [0,1] (audio SNR score, face-detected ratio…) | per sample, optional |
| `consistency_m` | 1 − mean Jensen–Shannon divergence to the other modalities | per sample |

Pooling is either `linear` (the weighted mean from the project brief) or `log` (weighted product of
experts, with uniform smoothing so a probability of 0 is not an absolute veto). `γ, β, λ` and the
pooling are chosen on a **dev stress set**, never on test. Results also carry `conflict`
(modalities disagree) and `uncertain` flags; conflicting evidence is reported, not hidden.

## Tests and experiment

```bash
python -m pytest fusion/tests -q                    # 35 tests (pytest is a dev dependency)
python -m fusion.experiments.synthetic_experiment   # ~2 min, writes fusion/results/synthetic_results.json
```

### What the synthetic experiment shows -- and does not show

It uses **simulated** model outputs (3 modalities of different strength, over-confidence, noisy
quality signals; scenarios: clean, noisy audio, each modality missing, random missing, 30 % injected
confident-wrong predictions). It checks the mechanics. **It is not evidence about MELD/MOSEI accuracy.**
Macro-F1, mean over 5 independent dev/test draws:

| scenario | equal weight | + calibration | reliability (linear) | reliability (log) |
|---|---|---|---|---|
| clean | 0.697 | 0.737 | 0.729 | **0.764** |
| noisy audio (quality known) | 0.556 | 0.624 | 0.622 | **0.640** |
| audio missing | 0.598 | 0.635 | 0.626 | **0.653** |
| text missing | 0.542 | 0.555 | 0.555 | **0.581** |
| 30 % conflicting | 0.573 | 0.614 | 0.626 | **0.655** |

Honest reading:

* Most of the gain over the plain average comes from **calibration** (ECE 0.07–0.16 → ≈0.02), then from **log pooling**.
* The tuner turned the per-sample confidence and quality weights **off** (γ = β = 0) and kept
  consistency (λ = 1). Adding confidence weighting on top of calibrated equal weights *hurt* in the
  ablation (e.g. clean 0.736 → 0.702). Consistency helps mainly in the conflict scenario (+0.014).
  Whether quality weighting helps depends on how informative the real quality signal is -- unknown until
  real audio-SNR / face-detection scores exist.
* The default `conflict` flag (threshold 0.25 bits) is **not discriminative** in the simulation
  (65 % of clean samples flagged vs 74 % in the conflict scenario). Fused *confidence* detects errors
  better (AUROC 0.77 vs 0.63 for the conflict score); abstaining on the 20 % least-confident fused
  predictions lifted accuracy 0.714 → 0.791. Re-tune the threshold on real dev data before using the flag.

## Known gaps for integration (as of the repo snapshot this was built against)

1. **Text model label mismatch**: `text_emotion` is trained on `dair-ai/emotion` (sadness, joy, love,
   anger, fear, surprise) -- no `neutral`/`disgust`, and not MELD data. `from_text_probs` folds `love`
   into `joy`, but that is a modelling decision to agree with the text owner, and the model gives
   `neutral` (≈47 % of MELD) probability 0.
2. **Face model**: not in the repo yet.
3. **Speech model**: reported MELD test accuracy is 23.2 % (macro-F1 0.19) -- weak, so it will get a low
   prior; check it is better than chance per class before relying on it.
4. **Real predictions are needed on the same MELD dev/test utterances** (needs raw audio/video, which are
   git-ignored): save per-utterance probabilities for all three models keyed by `sample_id` from the
   MELD manifests, then fit on dev and report on test with this same protocol.
