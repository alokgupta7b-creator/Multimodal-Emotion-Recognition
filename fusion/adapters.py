"""Adapters from the teammates' model outputs to the fusion input format.

These are duck-typed on purpose (no imports of ``speech_emotion`` / sklearn),
so the fusion package stays independent of the other modules.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .core import ModalityInput
from .labels import MELD_LABELS, remap_probs

# Label order of the text model trained on `dair-ai/emotion` (see text_emotion/predict.py).
# NOTE: this differs from MELD -- there is no neutral/disgust, and there is "love".
DAIR_EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")

# A modelling decision, not a fact: agree it with the text-model owner. "love" is
# folded into "joy"; MELD's "neutral" and "disgust" get probability 0 from this model.
DAIR_TO_MELD = {"love": "joy"}


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
    """Text-model probabilities (e.g. ``model.predict_proba``) -> MELD-ordered ``ModalityInput``.

    ``probs`` must follow ``src_labels`` order. For a scikit-learn pipeline,
    pass ``[model.classes_ mapped to names]`` accordingly (the repo's text model
    outputs integer classes 0..5 in the dair-ai order above).
    """
    out = remap_probs(probs, src_labels, MELD_LABELS, DAIR_TO_MELD if mapping is None else mapping)
    return ModalityInput(probs=out, quality=quality)
