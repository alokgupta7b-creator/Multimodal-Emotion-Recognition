# Robust Feature Fusion (`fusion/`)

Decision-level fusion of the text, speech and face emotion models. Owner: Alok.
Integration contract: see "Interface" below. Design of the text-model policy: [`DESIGN_partial_support.md`](DESIGN_partial_support.md).

> **Status.** Verified: the code, by unit tests on synthetic inputs. **Not verified: any result on real data.**
> No real model predictions exist in the repository yet (no face model, no text model file, no saved speech
> predictions), so every performance number below comes from a *simulator* and says nothing about MELD or CMU-MOSEI.

## Interface

All probability vectors are over the **7 MELD classes in alphabetical order**:
`anger, disgust, fear, joy, neutral, sadness, surprise` (`fusion.MELD_LABELS`; identical to
`speech_emotion.model.label_mapping.MELD_CLASS_NAMES`).

```python
from fusion import (ReliabilityAwareFuser, text_support_config, from_text_probs, from_ser_prediction)

# The text model cannot predict neutral/disgust, so its support is declared (log pooling is the
# recommended pooling for that case, see DESIGN_partial_support.md). Use ReliabilityAwareFuser() when
# every modality predicts all 7 classes.
fuser = ReliabilityAwareFuser(text_support_config())
fuser.fit({"text": P_text_dev, "audio": P_audio_dev, "face": P_face_dev}, y_dev)    # DEV split only, (N, 7) each

result = fuser.fuse({
    "text":  from_text_probs(text_probs_in_dair_order),              # NaN, not 0, at neutral/disgust
    "audio": from_ser_prediction(ser_pred, quality=0.8),             # optional quality score in [0, 1]
    "face":  None,                                                   # unavailable: skipped, never "neutral"
})
result.label, result.confidence, result.available, result.missing, result.weights,
result.status, result.warnings, result.conflict, result.uncertain, result.partial_only

batch = fuser.fuse_batch({"text": T, "audio": A, "face": F}, quality={"audio": q})   # (N, 7) arrays
batch.pred, batch.probs, batch.status          # status: (modality, sample) codes
batch.summary()                                # counts of why modalities were dropped
```

Contract:

* `fuse()` is a thin wrapper over `fuse_batch()`: **one** validation and fusion path, so single and batch results
  cannot disagree (tested on 300 deliberately corrupted samples).
* **Bad model outputs never crash fusion.** The modality is dropped for that sample and the reason is a status code:
  `ok`, `renormalized`, `missing` (None or an all-NaN row), `non_finite`, `negative`, `zero_sum`, `unsupported_mass`.
  Quality scores get `ok`, `unknown` (none supplied), `invalid` (NaN/inf, treated as unknown), `clipped`.
  `FusionConfig(strict=True)` raises instead.
* **Interface bugs raise `FusionInputError`** (a `ValueError`): unknown modality name, wrong vector length or row
  count, non-numeric quality, misaligned or invalid labels in `fit()`. Invalid options raise `FusionConfigError`.
* **No invented probabilities.** A sample with no usable modality has `label=None` and all-NaN `probs`. For
  a modality with partial class support, columns outside its support are NaN (never 0) and are ignored.
* `equal_weight_fusion(inputs)` is the reference baseline; `FusionConfig.equal_weight_baseline()` is its configuration.
* Adapters: `from_ser_prediction(SERPrediction)`, `from_text_probs(probs)`, `text_probs_from_model(model, texts)`.

## Method

Per sample, the weight of modality *m* is `w_m = prior_m * conf_m^gamma * quality_m^beta * consistency_m^lambda`,
renormalised over the **available** modalities. **By default every factor is off** (gamma = beta = lambda = 0, no
priors), i.e. calibrated equal weights: the measurements below did not show a consistent gain from any of them.

| factor | meaning | learned from |
|---|---|---|
| calibration | temperature scaling of each model's probabilities (`softmax(log p / T)`) | dev NLL, rows whose true class the model can predict |
| `prior_m` | dev macro-F1 of the modality | dev |
| `conf_m` | 1 - normalised entropy of the calibrated prediction | per sample |
| `quality_m` | external score in [0,1] (audio SNR, face-detected ratio, ...) | per sample, optional |
| `consistency_m` | 1 - mean Jensen-Shannon divergence to the other modalities | per sample |

Pooling is `linear` (weighted mean, the brief's formula) or `log` (weighted product of experts, with a little uniform
smoothing so one tiny probability is not an absolute veto). A modality that cannot predict every class (the text model)
only redistributes probability *inside* its classes and never changes P(neutral) or P(disgust): see the design note for
the alternatives that were rejected and what this gives up. Results also carry `conflict` and `uncertain` flags; the
conflict rule is **provisional** (see below).

## Configuration

Every tunable value lives in `FusionConfig` (`ReliabilityAwareFuser(FusionConfig(...))` or keyword overrides). `basis` says why a
default is what it is: `evidence` = chosen because of the measurements below (simulated), `assumption` = a modelling
choice with no supporting measurement, `engineering` = numerical safeguard. Nothing is tuned on real data.
`FusionConfig.to_dict()` saves the exact configuration with a result.

| option | default | basis | meaning and evidence |
|---|---|---|---|
| `labels` | MELD 7 classes, alphabetical | engineering | Class names; the order of every probability vector. Must match the models' output order. |
| `modalities` | `("text", "audio", "face")` | engineering | Modality names accepted as input keys. |
| `modality_support` | `{}` (all classes) | engineering | Per-modality list of the classes the model can actually predict. Modalities not listed support all classes. Columns outside the support are ignored, never read as 0. See fusion/DESIGN_partial_support.md. |
| `pooling` | `'linear'` | assumption | 'linear' = weighted mean of probabilities (project-brief formula). 'log' = weighted product of experts. Log scored higher on SIMULATED data (+0.023 mean macro-F1, better in 8 of 9 scenarios, never worse), but that simulation assumes independent errors, which favours log by construction; choose after correlated-error and real-dev results. |
| `use_calibration` | `True` | evidence | Temperature-scale each modality (fit on dev data only; needs fit(), an un-fitted fuser warns once and applies nothing). Switching it off cost 0.038 macro-F1 on average in the simulation (worse in 8 of 9 scenarios) -- but the simulated models are over-confident by construction, so real-data benefit is unverified. |
| `use_priors` | `False` | evidence | Multiply weights by each modality's dev macro-F1. Simulated: -0.004 mean macro-F1 with linear pooling (worse in 5 of 9 scenarios, best case +0.003) and -0.002 with log. F1 values from different class subsets are also not directly comparable. Revisit in the prior-correction stage. |
| `confidence_power` | `0.0` | evidence | Exponent on (1 - normalised entropy) of the calibrated prediction. 0 = off. Simulated: -0.016 (linear) and -0.012 (log) mean macro-F1, worse in 8 of 9 scenarios; entropy is not a validated measure of correctness. |
| `quality_power` | `0.0` | assumption | Exponent on an externally supplied quality score in [0,1] (audio SNR, face-detected ratio). 0 = off. Simulated: it helps only where the supplied score is informative (+0.015 linear / +0.031 log in the noisy-audio scenario with a known quality, 5/5 draws) and hurts otherwise (down to -0.013); mean -0.003 (linear), +0.001 (log). No real quality signal exists yet; opt in once one does and has been validated on dev data. |
| `consistency_power` | `0.0` | evidence | Exponent on agreement with the other modalities. 0 = off. Simulated: helps only under injected conflicts (+0.014 linear / +0.008 log), flat or slightly negative elsewhere (mean -0.002 / -0.003); cannot work with 2 modalities (weights come out equal). |
| `weight_floor` | `0.001` | engineering | Lower bound on each weight factor so no weight is exactly 0. |
| `smoothing` | `0.01` | assumption | Weight of a uniform distribution mixed into each modality before log pooling, so a tiny probability is not an absolute veto. Untuned. |
| `conflict_threshold` | `0.25` | assumption | Mean pairwise Jensen-Shannon divergence (bits) above which differing arg-maxes are flagged as a conflict. NOT justified: the flag was not discriminative in simulation. Slated for replacement by a dev-derived uncertainty/abstention rule (P1). |
| `min_confidence` | `0.0` | engineering | Fused confidence below this marks the result 'uncertain'. 0 = never. |
| `sum_tol` | `0.001` | engineering | Rows whose sum differs from 1 by more than this are flagged 'renormalized'. |
| `min_fit_samples` | `30` | engineering | Minimum number of usable dev rows needed to fit a modality's temperature and prior; with fewer, both stay at 1.0 and fit() warns. A floor against fitting noise, not a recommendation (a few hundred rows is the realistic minimum for a stable temperature). |
| `strict` | `False` | engineering | True: raise FusionInputError for invalid model outputs (NaN/negative/zero-sum rows, mass on a declared-unsupported class, non-finite or out-of-range quality) instead of dropping them with a status code. Missing modalities (None / all-NaN rows) are always allowed. |

## Tests and experiments

```bash
python -m pytest fusion/tests -q                      # fusion-specific unit tests (numpy + pytest only)
python -m fusion.experiments.synthetic_experiment     # ~2 min, writes fusion/results/synthetic_results.json
python -m fusion.experiments.isolated_options         # ~15 s, writes fusion/results/isolated_options.json
```

### Simulated results (not real-data performance)

The simulator makes three modalities of different strength and over-confidence, with noisy quality scores, and applies
failure scenarios. Its modalities make **independent errors, which favours log pooling by construction**. Temperatures are
fitted on a dev draw and scores are on a separate test draw. Macro-F1, mean of 5 independent draws:

| scenario | equal weight (baseline) | + calibration (**default**) | + calibration, log pooling |
|---|---|---|---|
| clean | 0.697 | 0.737 | 0.769 |
| noisy audio, quality score supplied | 0.556 | 0.624 | 0.639 |
| noisy audio, no quality score | 0.552 | 0.622 | 0.638 |
| audio missing | 0.598 | 0.635 | 0.659 |
| face missing | 0.621 | 0.667 | 0.697 |
| text missing | 0.542 | 0.555 | 0.583 |
| text only | 0.559 | 0.559 | 0.559 |
| 25 % of inputs randomly missing | 0.615 | 0.644 | 0.664 |
| 30 % confidently-wrong input | 0.573 | 0.614 | 0.655 |

Each option measured alone against its control (full table: `fusion/results/isolated_options.json`; mean macro-F1 change over the
9 scenarios, scenarios won / tied / lost):

| option | vs | mean change | won / tied / lost |
|---|---|---|---|
| log pooling | linear | +0.023 | 8 / 1 / 0 |
| no calibration | linear | -0.038 | 0 / 1 / 8 |
| priors | linear | -0.004 | 2 / 2 / 5 |
| confidence weighting | linear | -0.016 | 0 / 1 / 8 |
| quality weighting | linear | -0.003 | 2 / 1 / 6 (+0.015 where the score is informative) |
| consistency weighting | linear | -0.002 | 1 / 4 / 4 (+0.014 under injected conflicts) |

Honest reading:

* Calibration is the largest effect, but the simulated models are over-confident by construction. Whether the real models
  need it is unknown.
* Log pooling beats linear in every scenario with more than one modality, but it is exactly the case the simulator favours.
  The default stays `linear` until correlated errors and real dev predictions have been tested.
* No weighting factor shows a consistent gain, so all are off by default. The dev-stress tuner in `synthetic_experiment.py` does
  switch quality and consistency on, but that selection does not hold up when each factor is isolated.
* The default conflict rule is **not discriminative** in the simulation (flags 65 % of clean samples vs 74 % with injected
  conflicts), while fused confidence detects errors better (AUROC 0.76 vs 0.62). Abstaining on the 20 % least-confident
  predictions raised simulated accuracy 0.718 to 0.790. (These three figures are from the first of the 5 draws only.)
  The rule is to be replaced by a dev-derived one.

## Known gaps

1. **No real predictions.** Needed: per-utterance probabilities from all three models on the same MELD dev and test
   `sample_id`s, with the models trained on `train` only. Nothing here has been evaluated on them.
2. **Text model:** trained on `dair-ai/emotion` (6 classes, tweets), not MELD. The `love -> joy` merge is an assumption. Whether
   to keep the conservative policy or learn a map from text outputs to MELD classes on dev predictions cannot be decided
   without those predictions.
3. **Speech model:** reported MELD test accuracy 23.2 % (macro-F1 0.19). The repo's `speech_emotion/training/mosei_data.py`
   currently has a stray markdown fence on line 7 (a `SyntaxError`) that prevents importing the package; that file is not part of this module.
4. **Face model:** not in the repository yet.
5. **Open in this module:** training-prior double counting, correlated-error experiments, the conflict/abstention rule.
