"""Robust decision-level fusion for multimodal emotion recognition."""
from .adapters import (
    DAIR_EMOTION_LABELS, DAIR_TO_MELD, TEXT_SUPPORT, from_ser_prediction, from_text_probs,
    text_probs_from_model, meld_text_probs_from_model, text_support_config,
)
from .calibration import apply_temperature, expected_calibration_error, fit_temperature
from .config import FusionConfig
from .core import (
    BatchResult,
    FusionResult,
    ModalityInput,
    ReliabilityAwareFuser,
    equal_weight_fusion,
)
from .errors import FusionConfigError, FusionInputError
from .labels import MELD_LABELS, MODALITIES, remap_probs
from .metrics import accuracy, confusion_matrix, macro_f1, per_class_prf

__all__ = [
    "DAIR_EMOTION_LABELS", "DAIR_TO_MELD", "TEXT_SUPPORT", "from_ser_prediction", "from_text_probs",
    "text_probs_from_model", "meld_text_probs_from_model", "text_support_config",
    "BatchResult", "FusionConfig", "FusionConfigError", "FusionInputError", "FusionResult",
    "ModalityInput", "ReliabilityAwareFuser", "equal_weight_fusion", "MELD_LABELS", "MODALITIES",
    "remap_probs", "apply_temperature", "fit_temperature", "expected_calibration_error",
    "accuracy", "confusion_matrix", "macro_f1", "per_class_prf",
]
