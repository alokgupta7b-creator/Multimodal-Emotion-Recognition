"""
speech_emotion.model.label_mapping
==================================
Deterministic label encoding, decoding, and validation for the 7 MELD emotion classes:
    'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'

Key properties:
- Deterministic, sorted canonical class ordering.
- Bidirectional mapping between string emotion names and integer indices [0..6].
- Strict validation: unknown, missing, empty, or malformed labels raise ValueError.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union
import numpy as np

from speech_emotion.config import MELD_EMOTIONS

# Canonical deterministic class ordering (alphabetical, matching standard sklearn classes_)
MELD_CLASS_NAMES: Tuple[str, ...] = tuple(sorted(MELD_EMOTIONS))
NUM_MELD_CLASSES: int = len(MELD_CLASS_NAMES)  # 7


class MELDLabelEncoder:
    """
    Deterministic label encoder for the 7 MELD emotion classes.

    Classes (ordered index 0 to 6):
    0: 'anger'
    1: 'disgust'
    2: 'fear'
    3: 'joy'
    4: 'neutral'
    5: 'sadness'
    6: 'surprise'
    """

    def __init__(self, classes: Sequence[str] = MELD_CLASS_NAMES) -> None:
        self.classes: Tuple[str, ...] = tuple(str(c).strip().lower() for c in classes)
        if len(self.classes) != len(set(self.classes)):
            raise ValueError(f"Duplicate class names provided: {classes}")

        self._label_to_index: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self._index_to_label: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def encode_label(self, label: Union[str, Any]) -> int:
        """
        Encode a single string emotion label into its integer index [0..6].

        Raises
        ------
        ValueError:
            If label is not a string or not in the canonical class set.
        """
        if label is None:
            raise ValueError("Emotion label cannot be None")

        if not isinstance(label, (str, np.str_)):
            raise ValueError(f"Emotion label must be a string, got {type(label).__name__}: {label!r}")

        clean = str(label).strip().lower()
        if not clean:
            raise ValueError("Emotion label cannot be empty or whitespace")

        if clean not in self._label_to_index:
            raise ValueError(
                f"Unknown emotion label {label!r}. "
                f"Supported MELD emotions: {list(self.classes)}"
            )

        return self._label_to_index[clean]

    def decode_label(self, index: Union[int, np.integer]) -> str:
        """
        Decode an integer class index [0..6] back to its string emotion label.

        Raises
        ------
        ValueError:
            If index is not an integer or outside [0..num_classes - 1].
        """
        if index is None or isinstance(index, (float, np.floating)):
            raise ValueError(f"Class index must be an integer, got: {index!r}")

        try:
            idx = int(index)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid class index {index!r}: {err}") from err

        if idx not in self._index_to_label:
            raise ValueError(
                f"Class index {idx} out of bounds. Expected integer in [0, {len(self.classes) - 1}]."
            )

        return self._index_to_label[idx]

    def encode(self, labels: Iterable[str]) -> np.ndarray:
        """
        Encode a sequence of string labels into a 1D int64 numpy array.
        """
        return np.array([self.encode_label(y) for y in labels], dtype=np.int64)

    def decode(self, indices: Union[Sequence[int], np.ndarray]) -> List[str]:
        """
        Decode a sequence or array of integer indices back to string labels.
        """
        return [self.decode_label(i) for i in indices]

    def validate_labels(self, labels: Iterable[str]) -> List[str]:
        """
        Validate all labels and return normalized lowercase strings.
        """
        normalized: List[str] = []
        for i, lbl in enumerate(labels):
            try:
                self.encode_label(lbl)
                normalized.append(str(lbl).strip().lower())
            except ValueError as exc:
                raise ValueError(f"Label validation failed at index {i}: {exc}") from exc
        return normalized

    def __repr__(self) -> str:
        return f"MELDLabelEncoder(classes={list(self.classes)})"
