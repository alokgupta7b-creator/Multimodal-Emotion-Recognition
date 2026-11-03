"""Adapters from the teammates' model outputs to the fusion input format.

Duck-typed on purpose (no imports of ``speech_emotion`` or scikit-learn), so the fusion
package stays independent of the other modules and of their dependencies.

Text model policy (see ``fusion/DESIGN_partial_support.md``): the text model was trained
on ``dair-ai/emotion`` and cannot predict MELD's ``neutral`` or ``disgust``. Its output is
therefore NOT padded with zeros. Unsupported classes are emitted as NaN and declared in
``FusionConfig.modality_support`` (use ``text_support_config()``), so the fuser reads
the text vector only as a distribution *over the classes the model knows*.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .config import FusionConfig
from .core import ModalityInput
from .labels import MELD_LABELS, remap_probs

# Class order of the text model trained on `dair-ai/emotion` (class ids 0..5, see
# text_emotion/predict.py). There is no neutral/disgust, and there is "love".
DAIR_EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")

# MODELLING ASSUMPTION, not a fact: "love" is folded into "joy". Agree it with the text-model
# owner; "love" is not identical to MELD's "joy". Pass ``mapping=`` to use another rule
# (``{"love": None}`` drops it instead).
DAIR_TO_MELD = {"love": "joy"}

# MELD classes the text model can predict under the mapping above.
TEXT_SUPPORT = ("anger", "fear", "joy", "sadness", "surprise")


def text_support_config(**overrides) -> FusionConfig:
    """A ``FusionConfig`` declaring that the text model only covers ``TEXT_SUPPORT``.

    ``pooling`` defaults to ``"log"`` here (not the global default): with log pooling a text
    vector that is uniform over its classes is exactly the identity, with linear pooling it
    still flattens the distribution inside the supported classes
    (``fusion/DESIGN_partial_support.md``). Pass ``pooling="linear"`` to override.
    """
    overrides.setdefault("pooling", "log")
    return FusionConfig(modality_support={"text": TEXT_SUPPORT}, **overrides)


def from_ser_prediction(pred, quality: Optional[float] = None) -> ModalityInput:
    """``speech_emotion.model.predict.SERPrediction`` -> ``ModalityInput``.

    Uses ``pred.probability_array`` (shape (7,), alphabetical MELD order).
    """
    return ModalityInput(probs=np.asarray(pred.probability_array, dtype=float), quality=quality)


def from_text_probs(
    probs: Sequence[float],
    src_labels: Sequence[str] = DAIR_EMOTION_LABELS,
    mapping: Optional[Mapping[str, Optional[str]]] = None,
    quality: Optional[float] = None,
) -> ModalityInput:
    """Text-model probabilities (source label order) -> MELD-ordered ``ModalityInput``.

    Classes the text model cannot predict are NaN, never 0. Use together with
    ``text_support_config()``; if the fuser is NOT told about the support, the NaN
    makes it drop the text modality with status ``non_finite`` rather than silently
    treating NaN as zero evidence.
    """
    out = remap_probs(probs, src_labels, MELD_LABELS,
                      DAIR_TO_MELD if mapping is None else mapping, missing_fill=np.nan)
    return ModalityInput(probs=out, quality=quality)


def text_probs_from_model(model, texts: Sequence[str]) -> np.ndarray:
    """``(N, 7)`` MELD-ordered text probabilities (NaN for unsupported classes) from a
    scikit-learn-style model with ``predict_proba`` and ``classes_`` holding the integer
    ``dair-ai/emotion`` class ids. ``texts`` must already be preprocessed the way the text
    model was trained (``text_emotion.preprocessing.preprocess_text``); this function does
    not preprocess anything.
    """
    raw = np.asarray(model.predict_proba(list(texts)), dtype=float)
    classes = [int(c) for c in model.classes_]
    if sorted(classes) != list(range(len(DAIR_EMOTION_LABELS))):
        raise ValueError(f"unexpected text model classes_ {classes}; expected ids 0..{len(DAIR_EMOTION_LABELS) - 1}")
    ordered = raw[:, [classes.index(i) for i in range(len(DAIR_EMOTION_LABELS))]]
    return remap_probs(ordered, DAIR_EMOTION_LABELS, MELD_LABELS, DAIR_TO_MELD, missing_fill=np.nan)


def meld_text_probs_from_model(model, texts: Sequence[str]) -> np.ndarray:
    """``(N, 7)`` MELD-ordered probabilities from a text model trained on MELD itself
    (``text_emotion/train.py``: a scikit-learn pipeline whose ``classes_`` are the MELD emotion names).

    Columns are reordered by NAME into ``MELD_LABELS`` order. The model must expose exactly the 7 MELD
    classes: a missing or unknown class raises ``ValueError`` instead of being filled with 0 or NaN
    (use ``text_support_config`` / ``text_probs_from_model`` for a model that really lacks classes).
    ``texts`` must be the RAW utterance text: that pipeline is trained on raw text, so passing it
    ``preprocess_text`` output (as ``text_emotion/predict.py`` currently does) is a train/inference mismatch.
    """
    classes = [str(c).strip().lower() for c in model.classes_]
    if sorted(classes) != sorted(MELD_LABELS):
        raise ValueError(f"text model classes_ {classes} are not the 7 MELD classes {list(MELD_LABELS)}")
    raw = np.asarray(model.predict_proba(list(texts)), dtype=float)
    return raw[:, [classes.index(l) for l in MELD_LABELS]]
