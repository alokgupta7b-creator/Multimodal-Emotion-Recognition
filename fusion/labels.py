"""Emotion label definitions and label remapping helpers.

The label order here is the *shared contract* between the text, audio, face and
fusion modules: every probability vector passed to the fusion module must be
ordered exactly like ``MELD_LABELS`` (or like the ``labels`` you configure).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

# MELD's seven classes, in alphabetical order. Agree on this order with the
# text / speech / face model owners before integration.
MELD_LABELS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

MODALITIES: tuple[str, ...] = ("text", "audio", "face")


def remap_probs(
    probs: Sequence[float] | np.ndarray,
    src_labels: Sequence[str],
    dst_labels: Sequence[str],
    mapping: Optional[Mapping[str, Optional[str]]] = None,
    missing_fill: float = 0.0,
) -> np.ndarray:
    """Re-express a probability vector (or an ``(N, K)`` matrix) in another label set.

    ``mapping`` maps a source label to a destination label; a label that is not
    in ``mapping`` maps to the identically named destination label. Mapping a
    source label to ``None`` drops it. Mass from several source labels that map
    to one destination label is summed. The result is renormalised over the mass
    that survived.

    Destination labels that no source label maps to are filled with
    ``missing_fill`` (default 0.0 for backward compatibility). Use
    ``missing_fill=np.nan`` when the source model genuinely cannot predict those
    classes: a 0 would assert "certainly not this class", which the model never
    said. The fusion module reads NaN-filled columns as "not covered" only when
    the modality is declared in ``FusionConfig.modality_support``.

    Example (CMU-MOSEI "happiness" -> MELD "joy")::

        remap_probs(p, mosei_labels, MELD_LABELS, {"happiness": "joy"})

    Note: the mapping is a modelling decision (MOSEI has no "neutral" class and
    its scores are intensities, not probabilities), so it must be agreed with
    the dataset owner rather than assumed.
    """
    arr = np.asarray(probs, dtype=float)
    single = arr.ndim == 1
    arr = np.atleast_2d(arr)
    if arr.shape[1] != len(src_labels):
        raise ValueError(
            f"probs has {arr.shape[1]} columns but src_labels has {len(src_labels)}"
        )
    mapping = dict(mapping or {})
    dst_index = {lab: i for i, lab in enumerate(dst_labels)}
    out = np.zeros((arr.shape[0], len(dst_labels)))
    hit = np.zeros(len(dst_labels), dtype=bool)
    for j, src in enumerate(src_labels):
        dst = mapping.get(src, src)
        if dst is None:
            continue
        if dst not in dst_index:
            raise ValueError(f"source label {src!r} maps to unknown label {dst!r}")
        out[:, dst_index[dst]] += arr[:, j]
        hit[dst_index[dst]] = True
    totals = out[:, hit].sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("no probability mass left after remapping")
    out[:, hit] = out[:, hit] / totals
    out[:, ~hit] = missing_fill
    return out[0] if single else out
