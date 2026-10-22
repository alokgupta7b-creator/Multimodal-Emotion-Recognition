"""
speech_emotion.model.predict
============================
Clean prediction interface for Speech Emotion Recognition.

Capabilities:
-------------
1. Accepts 40-dimensional MFCC vectors (single vector or batch).
2. Performs strict validation on dimensions, NaN, and Inf values.
3. Returns structured `SERPrediction` containing predicted class, index, confidence,
   and deterministic class probabilities.
4. Provides convenience function to predict directly from an audio file.
5. Independent of MELD file paths for modular multimodal fusion integration.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from speech_emotion.config import AudioConfig, TOTAL_MFCC_FEATURES
from speech_emotion.feature_extraction import MFCCExtractor
from speech_emotion.model.classifier import SERClassifier
from speech_emotion.model.label_mapping import MELDLabelEncoder


@dataclasses.dataclass
class SERPrediction:
    """
    Structured prediction output for a single audio recording or feature vector.

    Attributes
    ----------
    predicted_label:
        Predicted emotion string ('anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise').
    predicted_index:
        Integer index [0..6] corresponding to the predicted class.
    confidence:
        Maximum probability value assigned to the top prediction (in [0.0, 1.0]).
    probabilities:
        Mapping from class name to probability score, ordered canonically.
    probability_array:
        1D numpy array of shape (7,) containing probabilities in canonical class order.
    """

    predicted_label: str
    predicted_index: int
    confidence: float
    probabilities: Dict[str, float]
    probability_array: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to JSON-serializable dictionary."""
        return {
            "predicted_label": self.predicted_label,
            "predicted_index": self.predicted_index,
            "confidence": float(self.confidence),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
        }


def predict_emotion(
    model: SERClassifier,
    features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
) -> Union[SERPrediction, List[SERPrediction]]:
    """
    Predict emotion from acoustic feature vector(s).

    Parameters
    ----------
    model:
        A fitted SERClassifier instance.
    features:
        1D array of shape (40,) for single sample, or 2D array of shape (N, 40) for batch.

    Returns
    -------
    SERPrediction if single sample, or List[SERPrediction] if batch.

    Raises
    ------
    ValueError:
        If features are not finite or feature dimensions != 40.
    RuntimeError:
        If model is not fitted.
    """
    if not isinstance(model, SERClassifier):
        raise TypeError(f"Expected SERClassifier instance, got {type(model).__name__}")

    # Validate features and determine single vs batch
    X_arr, is_single = model._validate_input_features(features)

    # Predict probabilities (N, 7) aligned with canonical classes
    probs = model.predict_proba(X_arr)
    classes = model.classes_

    predictions: List[SERPrediction] = []
    for i in range(len(X_arr)):
        p_row = probs[i]
        top_idx = int(np.argmax(p_row))
        top_label = classes[top_idx]
        conf = float(p_row[top_idx])

        prob_dict = {cls_name: float(p_row[c_i]) for c_i, cls_name in enumerate(classes)}

        pred = SERPrediction(
            predicted_label=top_label,
            predicted_index=top_idx,
            confidence=conf,
            probabilities=prob_dict,
            probability_array=p_row.copy(),
        )
        predictions.append(pred)

    if is_single:
        return predictions[0]
    return predictions


def predict_from_audio(
    model: SERClassifier,
    audio_path: Union[str, Path],
    config: Optional[AudioConfig] = None,
) -> SERPrediction:
    """
    Convenience function: load an audio file, extract 40 MFCC features, and predict emotion.

    Parameters
    ----------
    model:
        A fitted SERClassifier instance.
    audio_path:
        Path to an audio file on disk (.wav).
    config:
        Optional AudioConfig specifying sample_rate and extraction settings.

    Returns
    -------
    SERPrediction
    """
    path = Path(audio_path)
    extractor = MFCCExtractor(config or AudioConfig())

    features, meta = extractor.extract_from_file(path)
    if not meta.is_valid or features is None:
        raise ValueError(f"Cannot predict from invalid audio {path}: {meta.status} ({meta.reason})")

    res = predict_emotion(model, features)
    assert isinstance(res, SERPrediction)
    return res
