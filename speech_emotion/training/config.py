"""
speech_emotion.training.config
==============================
Configuration parameters for MELD SER model training, hyperparameter
tuning, and split evaluation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

from speech_emotion.config import MELD_EMOTIONS, TOTAL_MFCC_FEATURES
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


# Default repository root and data paths
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST_DIR = DEFAULT_REPO_ROOT / "datasets" / "MELD" / "manifests"
DEFAULT_CANDIDATE_FEATURE_DIR = Path.home() / "Documents" / "EmotionProject" / "features" / "mfcc"
DEFAULT_MODELS_DIR = DEFAULT_REPO_ROOT / "models" / "speech_emotion"


@dataclasses.dataclass
class DataConfig:
    """
    Configuration for loading MELD acoustic features and manifests.
    """
    manifest_dir: Path = DEFAULT_MANIFEST_DIR
    feature_dir: Path = DEFAULT_CANDIDATE_FEATURE_DIR
    expected_features: int = TOTAL_MFCC_FEATURES
    canonical_classes: Tuple[str, ...] = tuple(MELD_CLASS_NAMES)
    verify_split_isolation: bool = True

    def __post_init__(self) -> None:
        self.manifest_dir = Path(self.manifest_dir)
        self.feature_dir = Path(self.feature_dir)
        if self.expected_features <= 0:
            raise ValueError(f"expected_features must be positive, got {self.expected_features}")


@dataclasses.dataclass
class TrainingConfig:
    """
    Hyperparameter and execution configuration for training SER models.
    """
    model_family: str = "logistic_regression"  # 'logistic_regression', 'svm', 'mlp'
    standardize: bool = True
    class_weight: Optional[str] = "balanced"
    max_iter: int = 2000
    C: float = 1.0
    solver: str = "lbfgs"
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    mlp_hidden_layers: Tuple[int, ...] = (64, 32)
    mlp_alpha: float = 0.001
    mlp_early_stopping: bool = True
    random_state: int = 42
    output_dir: Path = DEFAULT_MODELS_DIR

    def __post_init__(self) -> None:
        allowed_families = ("logistic_regression", "svm", "mlp")
        if self.model_family not in allowed_families:
            raise ValueError(
                f"Unsupported model_family: {self.model_family!r}. Must be one of {allowed_families}"
            )
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.C <= 0.0:
            raise ValueError(f"C must be positive, got {self.C}")
        self.output_dir = Path(self.output_dir)
