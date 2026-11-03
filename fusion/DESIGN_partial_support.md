# Modalities that cannot predict every class (the text model)

Status: **implemented and unit-tested on synthetic inputs; not validated on real predictions.**

> **Update 2026-10-28.** The text model in `text_emotion/` was retrained on MELD and now predicts all 7 classes
> (string `classes_`; use `meld_text_probs_from_model`). For that model the partial-support machinery below is **not used**.
> It stays for any model that lacks classes, and the section describes the older 6-class `dair-ai/emotion` model.
> Class priors (separate topic) are handled in README section "Class priors".
Nothing in this document is a measured property of the real text model.

## 1. The problem

The text model (`text_emotion/`, TF-IDF + logistic regression) is trained on `dair-ai/emotion`:
`sadness, joy, love, anger, fear, surprise`. The fusion label space is MELD's seven classes. After the
mapping `love -> joy` (an assumption, see section 6) the text model can predict **five** MELD classes,
`S = {anger, fear, joy, sadness, surprise}`. It has **no output for `neutral` and `disgust`**.

Those two classes are not rare: they are **49.9 %** of MELD train (4,981 / 9,989), **44.4 %** of dev
(492 / 1,109) and **50.7 %** of test (1,324 / 2,610) (counts from `datasets/MELD/manifests/`).

What the text model outputs is `t(c) = P(c | x, c in S)`: a distribution over S *given that the answer is
in S*. It carries no statement about whether the answer is in S at all.

## 2. Alternatives and what each one asserts

| # | Policy | What it silently asserts | Verdict |
|---|---|---|---|
| A | **Zero-fill** the missing classes | "P(neutral) = 0": the model is *certain* the utterance is not neutral. Linear pooling: an implicit vote against neutral. Log pooling: a veto (the log of 0), only hidden by smoothing. | Rejected. The model never said this. |
| B | **Uniform / prior fill** (`1/7`, or MELD's class prior) | A made-up probability for classes the model cannot see; to keep a sum of 1 it also takes mass away from S by an arbitrary amount. | Rejected: invented numbers. |
| C | **Drop the classes** (fuse over S only) | That neutral/disgust do not exist, so the system can never output them although they are about half of MELD. Valid only as an *evaluation view* on the subset whose true class is in S. | Not used for prediction. |
| D | **Skip the term** (product of experts, text simply absent for neutral/disgust) | Looks like "no evidence", but a missing factor is worth `log 1 = 0`, higher than any present factor `log p <= 0`. It systematically favours exactly the classes the text model cannot see. | Rejected (biased). |
| E | **Likelihood-ratio form** (divide each model by its training prior, unsupported ratio = 1) | The sound generalisation of D, but it needs the text model's *training* class prior and the target prior. Without them it needs a placeholder prior. | Deferred to the prior-correction stage. |
| **F** | **Base-preserving factorisation** (implemented) | That text is informative only *inside* S and silent about S-versus-rest. | **Implemented.** |
| G | **Learned extension**: fit a small map from text outputs (log-probabilities, entropy, max) to the 7 classes on development predictions | Nothing invented; it can *use* text confidence as evidence about S-versus-rest. | **Cannot be built or judged without real dev predictions.** |

## 3. The implemented policy (F)

Declare the support once, in the configuration:

```python
from fusion import ReliabilityAwareFuser, text_support_config, from_text_probs
fuser = ReliabilityAwareFuser(text_support_config(pooling="log"))   # text covers TEXT_SUPPORT only
text = from_text_probs(model_probs_in_dair_order).probs               # NaN (not 0) at neutral, disgust
```

For a sample with at least one full-support modality (speech, face):

1. **Base.** Fuse the full-support modalities exactly as if no text existed: `P_base` over all 7 classes
   (same weights, calibration and pooling as the non-text path).
2. **Text updates inside S only.**

```
c not in S :  P(c) = P_base(c)                                   # untouched by text
c in S     :  P(c) = P_base(S) * q(c)
   linear  :  q = (W_b * P_base(.|S) + w_t * t) / (W_b + w_t)     # W_b = sum of full-support weights
   log     :  q  proportional to  P_base(.|S) * t^nu              # nu = w_t / mean full-support weight
```

Consequences (all asserted in `fusion/tests/test_fusion_p0.py`):

* `P(S)` and every `P(neutral)`, `P(disgust)` equal what speech+face alone give, **whatever the text
  says and however much weight it gets**. Text can only redistribute mass *within* S.
* **Log pooling:** a text vector that is uniform over S is exactly the identity (measured on 4,000
  random speech/face pairs: 0.0 % arg-max changes, identical probabilities).
* **Linear pooling:** a uniform text still flattens the peak inside S, because a linear pool treats
  every opinion as evidence. Measured on the same data: 8.4 % arg-max changes. (The first version of this
  policy, which re-pooled the classes separately, changed 16.4 % linear / 18.9 % log.)
  **Prefer `pooling="log"` whenever a partial-support modality is configured.**
* **Text only** (speech and face missing/invalid): the output is the text conditional on S, `NaN` outside S,
  `partial_only=True` and a warning. The system *cannot* answer neutral/disgust here; evaluate these
  rows separately.
* **Calibration** of the text model: temperature is fitted only on dev rows whose true class is in S, on
  the distribution restricted to S (`fit_report["text"]["n_used"]` says how many).
* **Validation:** the text model must declare its support. `NaN` at an undeclared position drops the modality
  (`non_finite`), it is never read as 0. Real mass (> `sum_tol`) at a declared-unsupported position drops the
  row (`unsupported_mass`; raises in `strict` mode). Zero placeholders are accepted.

Illustration (synthetic, one hand-made example): text says `joy` 0.8, speech and face lean `neutral`.
Zero-fill under log pooling answers *joy* with P(neutral)=0.07; the policy answers *neutral* with
P(neutral)=0.90, identical to speech+face alone.

## 4. What this policy gives up

The policy is conservative. It **discards whatever the text model's confidence says about neutral versus
emotional**. In real MELD text, flat/short utterances are probably informative about neutral, but this
model was never trained on neutral text, so its confidence on such inputs is out of distribution and
cannot be trusted as evidence. How much accuracy is lost by ignoring it is an empirical question
(alternative G) that **cannot be answered without real text predictions on MELD dev**.

Practical consequence: neutral/disgust are decided **only by the speech and face models**. The speech model
reports 23.2 % MELD test accuracy (macro-F1 0.19), so expect weak neutral detection until the face model
exists. This is a property of the available models, not something the fusion rule can repair.

## 5. What cannot be decided without real dev predictions

1. Whether policy F or a learned extension (G, validated with cross-validation grouped by `dialogue_id`)
   is better. G is only worth its complexity if it beats F on held-out dev dialogues.
2. Whether `love -> joy` is acceptable, or `love` should be dropped (`mapping={"love": None}`).
3. Whether linear or log pooling is better with real, correlated errors (simulation assumes independence,
   which favours log).
4. The size of the domain shift (tweets `dair-ai/emotion` -> dialogue lines) within S.

## 6. Assumptions and known limitations

* `t` is `P(c | x, c in S)`; true for a classifier trained only on S, not for other models.
* **Prior double counting (open):** `t` contains the text model's training class prior (dair is
  joy/sadness-heavy). The log update multiplies it in without dividing it out. Slated for the
  prior-correction stage (alternative E).
* `love -> joy` is a modelling assumption, not a fact.
* All partial-support modalities must share one supported set (`FusionConfigError` otherwise).
* `FusionResult.weights` for text is its strength *within S* only; it does not mean the same thing as the
  weight of a full-support modality.
* The conflict score is computed on the classes two modalities share; a pair with no shared mass is
  "not measurable" and contributes nothing. Conflict logic is provisional (to be replaced).
* Independence between modalities' errors is assumed by log pooling and is untested on real data.
