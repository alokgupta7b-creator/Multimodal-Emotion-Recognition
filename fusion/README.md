# Robust Feature Fusion (`fusion/`)

Decision-level fusion of the text, speech and face emotion models. Owner: Alok.
Integration contract: see "Interface" below. Design of the text-model policy: [`DESIGN_partial_support.md`](DESIGN_partial_support.md).

> **Status.** Verified: the code, by unit tests on synthetic inputs. Real predictions exist for **two of the three modalities**
> (speech: exported by the speech owner; text: my independent reproduction of the text owner's pipeline, not their saved file; face: none),
> so the only real-data fusion result is the preliminary text + speech check in "First check on real predictions". Every other performance
> number below comes from a *simulator* and says nothing about MELD or CMU-MOSEI.

## Interface

All probability vectors are over the **7 MELD classes in alphabetical order**:
`anger, disgust, fear, joy, neutral, sadness, surprise` (`fusion.MELD_LABELS`; identical to
`speech_emotion.model.label_mapping.MELD_CLASS_NAMES`).

```python
from fusion import (FusionConfig, ReliabilityAwareFuser, meld_text_probs_from_model, from_ser_prediction)

# Text model trained on MELD (text_emotion/train.py): all 7 classes, so no partial-support handling is needed.
# Declare the class prior each model was trained with (class_weight="balanced" -> "uniform"; unweighted ->
# its training class frequencies) so priors are not counted once per modality. See "Class priors" below.
fuser = ReliabilityAwareFuser(FusionConfig(pooling="log", model_priors={"text": "uniform"}))
fuser.fit({"text": P_text_dev, "audio": P_audio_dev, "face": P_face_dev}, y_dev)    # DEV split only, (N, 7) each

result = fuser.fuse({
    "text":  meld_text_probs_from_model(text_model, raw_utterances),   # (N, 7), RAW text, reordered by class name
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
* **Save / load.** `fuser.save("fuser.json")` writes the fitted state (config, temperatures, modality priors, fused temperature, dev class prior, fit
  report; no dev predictions) and `ReliabilityAwareFuser.load("fuser.json")` restores a fuser that gives identical outputs, so integration does
  not need the dev predictions. A malformed file raises `FusionInputError`. `state_dict()` / `from_state()` are the in-memory versions.
* `equal_weight_fusion(inputs)` is the reference baseline; `FusionConfig.equal_weight_baseline()` is its configuration.
* Adapters: `from_ser_prediction(SERPrediction)`; `meld_text_probs_from_model(model, texts)` for the MELD-trained text model
  (reorders by class name, raises if any of the 7 classes is missing, never fills one in). `from_text_probs` and
  `text_probs_from_model` are for the **older 6-class dair-ai/emotion text model** (NaN, not 0, at neutral/disgust) and are kept for it.

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

## Class priors (log pooling is a product of posteriors)

A model's posterior embeds the class prior of its training data: an unweighted logistic regression on MELD
carries neutral ~ 47 %, while `class_weight="balanced"` approximates a uniform prior. For conditionally independent
modalities Bayes gives

    P(c | x_1..x_M)  ~  pi_target(c) * prod_m  p_m(c | x_m) / pi_m(c)

so each model's own prior `pi_m` must be divided out and the target prior applied **once**. A plain product (the
previous behaviour) is exact only when every `pi_m` is uniform; otherwise it counts a non-uniform prior once per
available modality, so the implied prior changes with how many modalities are present (3 with face, 2 when it is missing).
Configure it with `model_priors` (what each model assumes; declared by its owner, never estimated here),
`target_prior` (`"uniform"`, `"dev"` or a vector) and `target_prior_power`. All defaults are uniform, which reproduces the
earlier results exactly. In `linear` pooling each expert is converted `p * pi_t / pi_m` before averaging (exact for one expert; a
mixture has no joint identity). A partial-support modality still never moves mass onto classes it cannot predict.

The target prior is a **trade-off, not a free gain**. On the real MELD text model alone (my independent reproduction of
`text_emotion/train.py`, text only, T fitted on dev, scored on test): uniform target accuracy 0.516 / macro-F1 0.378; dev class
frequencies 0.597 / 0.347; alpha = 0.5 gives 0.582 / 0.379. Pick it by the metric the project reports, on dev data only.

**Calibration under priors.** Scaling `log p` by 1/T also scales the prior term, which a later prior division cannot undo, and a
temperature fitted on natural-prior dev data absorbs the shift between a balanced model's training prior and the dev prior. When any prior
option is set, calibration therefore scales only the evidence (`log p - log pi_m`) and fits T against the dev class prior; with defaults
it is the original `softmax(log p / T)`. Real text-only outputs (independent reproduction, dev-fitted, test-scored, dev-frequency target):
balanced model ECE 0.035 -> 0.018, NLL 1.223 -> 1.213; unweighted model (declared training prior) ECE 0.068 -> 0.043, NLL 1.201 -> 1.194.
Accuracy is unchanged (0.595 -> 0.596, 0.600 -> 0.597) and macro-F1 moves by about 0.014 (0.347 -> 0.333, 0.350 -> 0.353), so this is a calibration
fix, not an accuracy gain. Whether it helps the fused output is untested until real audio and face predictions exist.

Simulated check (`prior_correction.py`, exact posteriors, macro-F1, all three modalities present, test labels at MELD frequencies):

| models trained with | no correction (before) | declared, uniform target | declared, natural target | declared wrongly |
|---|---|---|---|---|
| all balanced (class_weight=balanced) | 0.717 | 0.717 | 0.753 | 0.343 |
| all unweighted (natural prior) | 0.574 | 0.717 | 0.753 | 0.574 |
| mixed: text unweighted, audio/face balanced | 0.753 | 0.717 | 0.753 | 0.597 |

* The corrected fusion equals the exact Bayes posterior to 2e-15; this is an algebra check, not a performance claim.
* Without correction, models trained on the natural prior over-predict neutral more as modalities are added.
* **A wrong declaration is worse than no correction** (last column). `model_priors` must match how each model was really
  trained. Check on dev data: a calibrated model's mean predicted probability per class should be close to the prior it assumes.
  Dividing out a prior is only approximate for regularised models.

## Abstention threshold from dev data

`min_confidence` (abstain when the fused top probability is below it) should come from data, not a guess. `fusion/abstention.py`:

```python
oof = out_of_fold_confidence(cfg, dev_probs, y_dev, dialogue_ids, n_folds=5)      # refit on other dialogues, score the held-out ones
thr, info = select_min_confidence(oof["confidence"], oof["correct"], coverage=0.8)   # or max_risk=0.35
fuser = ReliabilityAwareFuser(cfg, min_confidence=thr)
```

Folds are by **dialogue**, so utterances of one dialogue never sit in both the fitting and the scoring part. Only dev data is used.
The conflict flag (`conflict_threshold`) is still the provisional rule and is included in `uncertain`; set `conflict_threshold=1.0`
to abstain on confidence alone.

Real text-only check (independent reproduction, balanced model, 1,109 dev utterances in 114 dialogues, 2,610 test): fused confidence
separates correct from wrong predictions with AUROC 0.67 on dev (out-of-fold) and 0.68 on test. A threshold chosen for 80 % dev coverage
kept 79.7 % of test utterances and lowered the test error among kept predictions from 0.484 to 0.428 (60 % coverage: 0.374). Choosing the
threshold from in-sample instead of out-of-fold scores gave the same thresholds to three decimals here, because only one temperature is fitted per
model; the grouped procedure matters when more is fitted on dev, and costs nothing. This is text only: nothing is known yet about the fused
output with audio and face.

## First check on real predictions (preliminary: 2 of 3 modalities)

Inputs: **speech** = `features/fusion/speech_{dev,test}_probabilities.csv` as pushed by the speech owner (1,105 of 1,109 dev and 2,606 of 2,610 test
utterances; the 4 missing per split are treated as missing modality); **text** = my independent reproduction of `text_emotion/train.py`
(not the text owner's saved output, which is not in the repository; their reported 51.46 % / macro-F1 0.38 matches my 0.516 / 0.378). No face model.
Run with `evaluate_real` (`--probs audio=... --strip-split-prefix`), everything chosen on dev, one test set (2,610 utterances), uncertainty by
paired bootstrap over test **dialogues** (1,000 resamples, 95 % interval), fused vs each model alone under the same configuration.

What the speech outputs are: accuracy 0.494 on test but macro-F1 only 0.131, because it labels 95 % of utterances neutral; its mean predicted
probabilities equal the natural class frequencies. It behaves like an unweighted model that mostly returns the class prior. That is why `model_priors` matters:

| fusion configuration (log or linear as selected on dev) | accuracy | macro-F1 | fused minus text alone: accuracy | fused minus text alone: macro-F1 |
|---|---|---|---|---|
| text alone | 0.516 | 0.378 | | |
| speech alone | 0.494 | 0.131 | | |
| priors NOT declared (plain pooling) | 0.591 | 0.367 | +0.076 [+0.059, +0.093] | -0.011 [-0.031, +0.009] |
| speech prior declared (train frequencies), uniform target | 0.526 | 0.383 | +0.011 [-0.002, +0.023] | +0.005 [-0.008, +0.017] |
| speech prior declared, dev-frequency target | 0.595 | 0.343 | -0.001 [-0.012, +0.011] | +0.011 [-0.001, +0.023] |

* The **+7.6 accuracy points** of plain pooling are the class prior, not information from the voice: the speech model re-supplies the neutral-heavy
  prior (text alone with the dev-frequency target reaches the same 0.596 accuracy). Macro-F1 did not improve. Without the prior correction this
  would have looked like a fusion win.
* With the priors handled correctly, **adding this speech model to text gives no detectable gain**: every interval includes 0. After the prior is
  divided out the speech model alone scores 0.262 accuracy / 0.199 macro-F1, i.e. weak but above chance, and the gain over text is within noise.
* The speech model's training prior is **inferred** from its outputs (mean probabilities, 95 % neutral predictions), not confirmed by its owner; if it
  was trained with class weights the declaration is wrong and these rows change.
* The conflict flag was rejected on dev (disagreement does not predict errors) and abstention at 80 % coverage worked on test (error among kept
  0.359-0.414 vs 0.405-0.474 overall, depending on the configuration).
* Limits: two modalities; one test set; the text side is a reproduction; the speech file's training protocol (train only) is as stated by its owner, not verified here.

## Evaluating real predictions

`python -m fusion.experiments.evaluate_real --data DIR [--config cfg.json]` evaluates on prediction files and invents nothing. Files in `DIR`
(CSV with header, written by the model owners from models trained on `train` only):

* `labels_dev.csv`, `labels_test.csv`: `sample_id,label,dialogue_id` (label is a MELD class name)
* `probs_<dev|test>_<text|audio|face>.csv`: `sample_id,anger,disgust,fear,joy,neutral,sadness,surprise` (columns matched by name)

A sample absent from a model's file counts as that modality being unavailable for it; an unknown or duplicate id, a missing or extra column,
a non-numeric cell, a modality with only one split, or no prediction files at all is an error (exit code 2, no output written).
Everything is chosen on dev (candidate by dialogue-grouped out-of-fold macro-F1; abstention threshold; conflict-flag decision) and one frozen choice
is reported on test: each modality alone, equal-weight baseline, all candidates, the selected configuration for every subset of modalities
available at test time, abstention, per-class precision/recall inputs and confusion matrices, plus SHA-256 and row counts of every input file.
It has been run on real **text-only** outputs (reproduces the numbers above) and its mechanics are unit-tested on synthetic files; it has
**not** been run on real multi-modal predictions because none exist yet. It cannot verify that the files really come from the stated models.

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
| `model_priors` | `{}` | assumption | Per-modality class prior that the model's probabilities ASSUME (the prior of its training distribution): 'uniform' (default) or one positive number per label, in label order. class_weight='balanced' training -> uniform; unweighted training -> the training class frequencies. It is divided out before pooling so a prior is not counted once per model (Bayes: P(c|x_1..x_M) ~ pi_t(c) * prod_m p_m(c|x_m) / pi_m(c)). Declared by the model owner, never estimated here; a wrong declaration makes results worse, not better. Partial-support modalities must stay 'uniform'. |
| `target_prior` | `'uniform'` | assumption | Class prior of the FUSED output, applied exactly once: 'uniform' (default: no prior imposed; suits macro-F1 / balanced accuracy), 'dev' (class frequencies of the y_dev given to fit(), Laplace-smoothed; suits accuracy) or an explicit vector. This is a real accuracy / macro-F1 trade-off, not a free gain: on the real MELD text model alone (independent reproduction, text only) moving uniform -> dev frequencies raised test accuracy 0.516 -> 0.597 and lowered test macro-F1 0.378 -> 0.347. |
| `target_prior_power` | `1.0` | assumption | Exponent alpha in [0, 1] applied to a non-uniform target prior (pi_t ~ pi^alpha): 0 = uniform, 1 = as given. Interpolates between the two objectives (text-only check on real MELD outputs: alpha=0.5 gave test accuracy 0.582 / macro-F1 0.379 vs 0.516 / 0.378 for uniform). Choose it on dev data only. Ignored when target_prior is 'uniform'. Setting any prior option also switches calibration to the prior-aware form: only the evidence log p - log pi_m is scaled by 1/T, and T is fitted against the dev class prior. |
| `calibrate_fused` | `False` | evidence | Fit one extra temperature on the FUSED dev output (after pooling) and apply it to every fused output. Argmax is unchanged; only confidence changes. Calibrating each model separately does not make the pooled probabilities calibrated when modality errors are correlated: in simulation with shared noise (rho=0.6) log-pooled NLL was 1.29 vs 1.09 for linear, and the fused temperature recovers most of that (see results/correlated_errors.json). One temperature is fitted on all-modalities-present dev rows, so it is only approximately right when modalities are missing at run time. Needs fit(). |
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
python -m fusion.experiments.prior_correction         # a few seconds, writes fusion/results/prior_correction.json
python -m fusion.experiments.correlated_errors        # ~20 s, writes fusion/results/correlated_errors.json
python -m fusion.experiments.conflict_rule            # ~15 s, writes fusion/results/conflict_rule.json
python -m fusion.experiments.evaluate_real --data DIR # real predictions from files, see "Evaluating real predictions"
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
* Log pooling beats linear in every scenario with more than one modality, but only because this simulator's errors are independent.
  With correlated errors that advantage disappears (next section). The default stays `linear`.
* No weighting factor shows a consistent gain, so all are off by default. The dev-stress tuner in `synthetic_experiment.py` does
  switch quality and consistency on, but that selection does not hold up when each factor is isolated.
* The fixed default conflict rule is **not discriminative** in the simulation (flags 65 % of clean samples vs 74 % with injected
  conflicts), while fused confidence detects errors better (AUROC 0.76 vs 0.62). Abstaining on the 20 % least-confident
  predictions raised simulated accuracy 0.718 to 0.790. Both are now chosen from dev data, see "Abstention" and "Conflict flag".

### Correlated errors (simulated, not real-data performance)

Real modalities share errors, so `correlated_errors.py` adds a noise term common to all modalities (`rho` = its share) and a shared
"confuser" class that pushes every modality to the same wrong answer. Cells are macro-F1 / NLL of the fused output, mean of 5 draws,
calibration fitted on a dev draw of the same process; `fusedT` is `calibrate_fused=True` (argmax, hence macro-F1, is unchanged by it).

| setting (simulated) | linear | log | linear + fusedT | log + fusedT |
|---|---|---|---|---|
| independent errors (rho=0) | 0.811 / 0.98 | 0.840 / 0.47 | 0.811 / 0.56 | 0.840 / 0.47 |
| shared noise rho=0.3 | 0.717 / 1.03 | 0.721 / 0.87 | 0.717 / 0.82 | 0.721 / 0.79 |
| shared noise rho=0.6 | 0.646 / 1.09 | 0.637 / 1.29 | 0.646 / 1.01 | 0.637 / 1.01 |
| shared noise rho=0.9 | 0.589 / 1.17 | 0.577 / 1.70 | 0.589 / 1.16 | 0.577 / 1.17 |
| shared confuser, 20% of samples | 0.592 / 1.32 | 0.589 / 1.52 | 0.592 / 1.24 | 0.589 / 1.25 |
| shared confuser, 40% of samples | 0.467 / 1.50 | 0.461 / 1.83 | 0.467 / 1.46 | 0.461 / 1.48 |
| rho=0.6, audio signal x0.5 at test | 0.577 / 1.24 | 0.556 / 1.66 | 0.577 / 1.21 | 0.556 / 1.23 |
| rho=0.6, audio signal x0.0 at test | 0.528 / 1.35 | 0.475 / 2.10 | 0.528 / 1.36 | 0.475 / 1.48 |

* Log pooling's advantage (+0.029 macro-F1 at rho=0) shrinks to +0.004 at rho=0.3 and reverses from rho=0.6 on (-0.009 to -0.012); with a degraded modality that
  calibration did not anticipate, linear is 0.02 to 0.05 better. A product of experts counts shared evidence more than once.
* Under correlation log pooling is also **over-confident** (NLL 1.29 vs 1.09 at rho=0.6), which would distort confidence-based
  abstention. One temperature fitted on the fused dev output repairs most of it (log 1.29 -> 1.01) and also helps linear pooling.
  It is opt-in (`calibrate_fused`) because the evidence is simulated.
* So the defaults (`linear`, per-model calibration) stay; `log` is only justified if real dev predictions show conditionally
  independent errors, which `evaluate_real` lets the team check (out-of-fold macro-F1 selects the candidate).

### Conflict flag

The fixed default (`conflict_threshold=0.25`) is a placeholder. `select_conflict_threshold` (in `fusion/abstention.py`) enables the
flag only if, on out-of-fold dev data, the most-disagreeing 20 % of rows with two or more modalities are wrong at least 1.2x as
often as the rest and a one-sided z-test reaches 5 %; otherwise it returns `None` and the flag should stay off.

| setting (simulated) | dev rule accepts the flag (of 5 draws) | mean lift | default 0.25 rule: rows flagged | error flagged / unflagged |
|---|---|---|---|---|
| independent errors | 5 | 1.48 | 68% | 0.23 / 0.12 |
| shared noise rho=0.6 | 0 | 0.95 | 15% | 0.32 / 0.36 |
| one modality misleading on 25% of rows | 0 | 1.07 | 65% | 0.25 / 0.21 |
| shared confuser 30% | 0 | 0.74 | 2% | 0.26 / 0.47 |

Disagreement signals error only when errors are independent; when modalities agree on the same wrong answer it is uninformative or
even inverted (lift 0.74 with a shared confuser), and a single corrupted modality that the others outvote does not make the *fused*
answer wrong, so the flag does not detect it. The rule therefore says "off" where the flag would mislead. The default threshold was
not changed, to avoid silently altering the integrated output.

## Known gaps

1. **Real predictions are incomplete.** Speech outputs exist (see "First check on real predictions"); the face model has none and the text
   owner's own probability file is not in the repository. Needed: all three on the same MELD dev and test `sample_id`s, trained on `train` only.
2. **Text model:** `text_emotion/` now trains on MELD (7 classes; reported ~51 % accuracy). I reproduced that independently
   (text only) but no prediction file from the team's own run is in the repository, so it is not verified from their artifact.
   `text_emotion/predict.py` preprocesses text while `train.py` / `evaluate.py` use raw text, and the README still quotes an old
   41.38 %: both are the text owner's to fix. The partial-support policy remains for any model that lacks classes.
3. **Speech model:** the exported probabilities are close to the class prior (95 % of test utterances labelled neutral, macro-F1 0.13; 0.20 after
   the prior is divided out), so it adds little to text so far. Its training prior is inferred, not confirmed. The earlier syntax error in
   `mosei_data.py` was fixed by its owner.
4. **Face model:** not in the repository yet.
5. **Untested with all three modalities:** the comparisons above use text + speech only. The choice between linear and log pooling,
   `calibrate_fused`, reliability weighting under real noise, and the conflict flag with three modalities still need the face predictions.
