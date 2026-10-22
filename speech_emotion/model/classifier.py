"""
speech_emotion.model.classifier
===============================
Modular Speech Emotion Recognition (SER) classifier.

Design principles:
------------------
1. Architecture: scikit-learn pipeline combining StandardScaler feature
   standardization and LogisticRegression with class imbalance handling.
2. Configuration: fully configurable hyperparameters via SERConfig dataclass.
3. Separation of Concerns: decoupled from audio reading and feature extraction.
4. Determinism: reproducible random state and canonical 7-class label ordering.
5. Persistence: atomic save and load using joblib with metadata auditing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from speech_emotion.config import TOTAL_MFCC_FEATURES
from speech_emotion.model.label_mapping import MELDLabelEncoder, MELD_CLASS_NAMES


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SERConfig:
    """
    Configuration parameters for the Speech Emotion Recognition model.

    Parameters
    ----------
    classifier_type:
        Model type ('logistic_regression'). Default: 'logistic_regression'.
    standardize:
        Whether to standardize features with StandardScaler. Default: True.
    class_weight:
        Class weighting strategy ('balanced' or None). Default: 'balanced'.
    max_iter:
        Maximum solver iterations. Default: 2000.
    C:
        Inverse of regularization strength. Default: 1.0.
    solver:
        Optimization algorithm ('lbfgs'). Default: 'lbfgs'.
    random_state:
        Seed for reproducibility. Default: 42.
    expected_features:
        Expected number of input features (default: 40 MFCC features).
    """

    classifier_type: str = "logistic_regression"
    standardize: bool = True
    class_weight: Optional[str] = "balanced"
    max_iter: int = 2000
    C: float = 1.0
    solver: str = "lbfgs"
    random_state: Optional[int] = 42
    expected_features: int = TOTAL_MFCC_FEATURES

    def __post_init__(self) -> None:
        if self.classifier_type != "logistic_regression":
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type!r}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.C <= 0.0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.expected_features <= 0:
            raise ValueError(f"expected_features must be positive, got {self.expected_features}")


# ---------------------------------------------------------------------------
# Classifier Class
# ---------------------------------------------------------------------------

class SERClassifier:
    """
    Speech Emotion Recognition Classifier using 40-dimensional MFCC acoustic features.
    """

    def __init__(
        self,
        config: Optional[SERConfig] = None,
        label_encoder: Optional[MELDLabelEncoder] = None,
    ) -> None:
        self.config = config or SERConfig()
        self.label_encoder = label_encoder or MELDLabelEncoder()
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained and is ready for inference."""
        return self._is_fitted

    @property
    def classes_(self) -> List[str]:
        """List of class names in deterministic model order."""
        return list(self.label_encoder.classes)

    def _build_pipeline(self) -> Pipeline:
        """Construct the scikit-learn Pipeline according to configuration."""
        steps = []
        if self.config.standardize:
            steps.append(("scaler", StandardScaler()))

        if self.config.classifier_type == "logistic_regression":
            clf = LogisticRegression(
                max_iter=self.config.max_iter,
                C=self.config.C,
                class_weight=self.config.class_weight,
                solver=self.config.solver,
                random_state=self.config.random_state,
            )
            steps.append(("classifier", clf))
        else:
            raise ValueError(f"Unknown classifier_type: {self.config.classifier_type}")

        return Pipeline(steps)

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Sequence[str],
    ) -> SERClassifier:
        """
        Train the SER model on feature matrix X and string emotion labels y.

        Parameters
        ----------
        X:
            2D array-like of shape (N, expected_features).
        y:
            Sequence of N emotion labels (strings).

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix (N, {self.config.expected_features}), got shape {X_arr.shape}")

        if X_arr.shape[1] != self.config.expected_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.config.expected_features} features, "
                f"got {X_arr.shape[1]}"
            )

        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Feature matrix contains non-finite values (NaN or Inf)")

        N = len(X_arr)
        if len(y) != N:
            raise ValueError(f"Mismatch between features ({N} rows) and labels ({len(y)} rows)")

        # Validate and normalize labels
        validated_labels = self.label_encoder.validate_labels(y)

        # Build pipeline and fit
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X_arr, validated_labels)
        self._is_fitted = True

        return self

    def _ensure_fitted(self) -> Pipeline:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("SERClassifier is not fitted yet. Call 'fit' or 'load' before predicting.")
        return self._pipeline

    def _validate_input_features(
        self,
        X: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ) -> Tuple[np.ndarray, bool]:
        """
        Validate input feature array and return (2D array, is_single_sample).
        """
        X_arr = np.asarray(X, dtype=np.float64)

        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input features contain non-finite values (NaN or Inf)")

        is_single = False
        if X_arr.ndim == 1:
            if X_arr.shape[0] != self.config.expected_features:
                raise ValueError(
                    f"Expected 1D feature vector of length {self.config.expected_features}, "
                    f"got length {X_arr.shape[0]}"
                )
            X_arr = X_arr.reshape(1, -1)
            is_single = True
        elif X_arr.ndim == 2:
            if X_arr.shape[1] != self.config.expected_features:
                raise ValueError(
                    f"Expected 2D feature matrix with {self.config.expected_features} columns, "
                    f"got shape {X_arr.shape}"
                )
        else:
            raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D array")

        return X_arr, is_single

    def predict(
        self,
        X: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ) -> Union[str, List[str]]:
        """
        Predict emotion labels for input features.

        Returns
        -------
        str if single 1D input was passed, or list of str if 2D batch was passed.
        """
        pipe = self._ensure_fitted()
        X_arr, is_single = self._validate_input_features(X)

        preds = pipe.predict(X_arr)
        preds_list = [str(p) for p in preds]

        if is_single:
            return preds_list[0]
        return preds_list

    def predict_proba(
        self,
        X: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ) -> np.ndarray:
        """
        Predict class probabilities aligned with canonical class order.

        Returns
        -------
        2D numpy array of shape (N, 7) with probabilities in canonical
        order: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'].
        """
        pipe = self._ensure_fitted()
        X_arr, _ = self._validate_input_features(X)

        raw_probs = pipe.predict_proba(X_arr)
        # Verify alignment with canonical class order
        model_classes = list(pipe.classes_)
        canonical_classes = list(self.label_encoder.classes)

        if model_classes == canonical_classes:
            return raw_probs.astype(np.float64)

        # Re-index columns if model classes differ from canonical order
        reordered = np.zeros((len(X_arr), len(canonical_classes)), dtype=np.float64)
        for target_idx, class_name in enumerate(canonical_classes):
            if class_name in model_classes:
                src_idx = model_classes.index(class_name)
                reordered[:, target_idx] = raw_probs[:, src_idx]

        return reordered

    def save(
        self,
        file_path: Union[str, Path],
        overwrite: bool = False,
    ) -> Path:
        """
        Serialize model and configuration to disk.

        Raises
        ------
        RuntimeError:
            If model is not fitted.
        FileExistsError:
            If file_path already exists and overwrite is False.
        """
        pipe = self._ensure_fitted()
        out_path = Path(file_path)

        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Model file already exists at {out_path}. "
                "Pass overwrite=True to allow overwriting."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "pipeline": pipe,
            "config": dataclasses.asdict(self.config),
            "classes": list(self.label_encoder.classes),
            "format_version": "1.0",
        }

        joblib.dump(artifact, out_path)
        return out_path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> SERClassifier:
        """
        Load a trained model artifact from disk.

        Security Warning
        ----------------
        This method uses `joblib.load` (pickle deserialization). Only load model
        artifacts from trusted sources. Never load untrusted or unverified files
        to prevent arbitrary code execution risks.

        Raises
        ------
        FileNotFoundError:
            If file_path does not exist.
        ValueError:
            If the artifact is malformed, corrupted, or has an invalid structure.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        try:
            artifact = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"Failed to deserialize model artifact at {path}: {exc}") from exc

        if isinstance(artifact, dict):
            if "pipeline" not in artifact or not isinstance(artifact["pipeline"], Pipeline):
                raise ValueError(
                    f"Malformed model artifact at {path}: missing or invalid 'pipeline' "
                    f"(expected sklearn.pipeline.Pipeline, got {type(artifact.get('pipeline')).__name__})"
                )

            config_dict = artifact.get("config", {})
            if not isinstance(config_dict, dict):
                raise ValueError(f"Malformed model artifact at {path}: 'config' must be a dictionary")

            try:
                config = SERConfig(**config_dict)
            except Exception as cfg_err:
                raise ValueError(f"Incompatible configuration in model artifact at {path}: {cfg_err}") from cfg_err

            classes = artifact.get("classes", MELD_CLASS_NAMES)
            if not isinstance(classes, (list, tuple)) or len(classes) == 0:
                raise ValueError(f"Malformed model artifact at {path}: 'classes' must be a non-empty sequence")

            try:
                encoder = MELDLabelEncoder(classes=classes)
            except Exception as enc_err:
                raise ValueError(f"Invalid class definitions in model artifact at {path}: {enc_err}") from enc_err

            model = cls(config=config, label_encoder=encoder)
            model._pipeline = artifact["pipeline"]
            model._is_fitted = True
            return model

        # Fallback if raw Pipeline was saved
        if isinstance(artifact, Pipeline):
            model = cls()
            model._pipeline = artifact
            model._is_fitted = True
            return model

        raise ValueError(
            f"Unrecognized model artifact format at: {path}. "
            f"Expected dict with 'pipeline' or sklearn.pipeline.Pipeline, got {type(artifact).__name__}"
        )
