"""Candidate 4: Text-Anchored Class-Conditional Bounded Multimodal Fusion.

Reproduces the winning Phase 13 fusion architecture:
1. Preserves text dominance when text predicts nuanced/rare emotions (fear, disgust,
   sadness, joy), preventing auxiliary dilution.
2. Selectively injects speech when vocal prosody demonstrates high-confidence acoustic
   activation for anger or surprise (acoustic margin over neutral > tau_audio).
3. Keeps video outside the default production prediction path (lambda_video = 0.0).
4. Handles missing or excluded modalities dynamically without synthetic zero/neutral
   probability vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from speech_emotion.model.label_mapping import MELD_CLASS_NAMES

CLASS_NAMES: tuple[str, ...] = tuple(MELD_CLASS_NAMES)
NUM_CLASSES: int = len(CLASS_NAMES)
FUSION_METHOD_IDENTIFIER: str = "text_anchored_class_conditional_bounded_fusion_v1"

# Empirically tuned hyperparameters from Phase 13 5-fold cross-validation
DEFAULT_TAU_AUDIO: float = 0.20
DEFAULT_LAMBDA_MAX: float = 0.40
DEFAULT_KAPPA_AUDIO: float = 1.50
DEFAULT_LAMBDA_VIDEO: float = 0.00

# Classes where auxiliary modalities must not override text predictions
PROTECTED_CLASSES: frozenset[str] = frozenset({"fear", "disgust", "sadness", "joy"})
SYNERGY_CLASSES: frozenset[str] = frozenset({"anger", "surprise"})


@dataclass(frozen=True)
class FusionConfig:
    tau_audio: float = DEFAULT_TAU_AUDIO
    lambda_max: float = DEFAULT_LAMBDA_MAX
    kappa_audio: float = DEFAULT_KAPPA_AUDIO
    lambda_video: float = DEFAULT_LAMBDA_VIDEO
    fusion_identifier: str = FUSION_METHOD_IDENTIFIER


@dataclass(frozen=True)
class FusionResult:
    predicted_emotion: str
    probabilities: dict[str, float]
    confidence: float
    fusion_method: str
    modalities_available: dict[str, bool]
    exclusion_reasons: dict[str, str | None]
    modality_weights: dict[str, float]
    rule_applied: str
    margin_audio: float | None = None


def _validate_probabilities(probs: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate and convert probability mapping/vector to canonical normalized float64 array."""
    if isinstance(probs, Mapping):
        missing = set(CLASS_NAMES) - set(probs.keys())
        if missing:
            raise ValueError(f"Missing classes in probability mapping: {sorted(missing)}")
        vec = np.array([float(probs[c]) for c in CLASS_NAMES], dtype=np.float64)
    else:
        vec = np.asarray(probs, dtype=np.float64).flatten()
        if len(vec) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} class probabilities, got {len(vec)}")

    if not np.isfinite(vec).all() or (vec < 0).any():
        raise ValueError("Probabilities must be finite non-negative numbers.")
    total = float(vec.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError(f"Probabilities must sum to 1.0 (got {total:.6f})")
    return vec / total


class Candidate4FusionEngine:
    """Production late fusion engine implementing Candidate 4."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._idx_anger = CLASS_NAMES.index("anger")
        self._idx_surprise = CLASS_NAMES.index("surprise")
        self._idx_neutral = CLASS_NAMES.index("neutral")
        self._protected_indices = frozenset(CLASS_NAMES.index(c) for c in PROTECTED_CLASSES)

    def fuse(
        self,
        text_probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
        audio_probabilities: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        video_probabilities: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        audio_exclusion_reason: str | None = None,
        video_exclusion_reason: str | None = None,
        video_quality: dict[str, Any] | None = None,
    ) -> FusionResult:
        p_t = _validate_probabilities(text_probabilities)

        avail_a = audio_probabilities is not None
        p_a: np.ndarray | None = None
        if avail_a:
            p_a = _validate_probabilities(audio_probabilities)

        avail_v = video_probabilities is not None and self.config.lambda_video > 0.0
        p_v: np.ndarray | None = None
        if avail_v:
            p_v = _validate_probabilities(video_probabilities)

        modalities_available = {
            "text": True,
            "audio": bool(avail_a),
            "video": bool(avail_v),
        }
        exclusion_reasons = {
            "audio": audio_exclusion_reason if not avail_a else None,
            "video": video_exclusion_reason or ("outside_production_path" if not avail_v else None),
        }

        pred_t_idx = int(p_t.argmax())
        pred_t_class = CLASS_NAMES[pred_t_idx]

        modality_weights = {"text": 1.0, "audio": 0.0, "video": 0.0}
        p_fused = np.copy(p_t)
        margin_audio: float | None = None

        # Rule 1: Text-Anchored Protected Class Safeguard
        if pred_t_idx in self._protected_indices:
            rule_applied = f"protected_class_lockout_{pred_t_class}"
            p_fused = p_t

        else:
            rule_applied = "text_anchor_default"
            # Rule 2: Acoustic synergy on anger or surprise
            if avail_a and p_a is not None:
                pred_a_idx = int(p_a.argmax())
                pred_a_class = CLASS_NAMES[pred_a_idx]
                if pred_a_class in SYNERGY_CLASSES:
                    # In Candidate 4 (Phase 13), acoustic margin M_a is defined specifically as the contrast
                    # between the synergy emotion (anger or surprise) and the neutral conversational baseline:
                    #   M_a = p_a[pred_a] - p_a[neutral]
                    # This measures deviation from emotionally neutral speech rather than raw top1 - top2 margin.
                    m_a = float(p_a[pred_a_idx] - p_a[self._idx_neutral])
                    margin_audio = m_a
                    if m_a > self.config.tau_audio:
                        lam_a = min(self.config.lambda_max, self.config.kappa_audio * m_a)
                        p_fused = (1.0 - lam_a) * p_fused + lam_a * p_a
                        modality_weights["text"] = float(1.0 - lam_a)
                        modality_weights["audio"] = float(lam_a)
                        rule_applied = f"acoustic_synergy_{pred_a_class}"

            # Rule 3: Optional Video Quality Gating (disabled by default in production)
            if avail_v and p_v is not None and self.config.lambda_video > 0.0:
                cov = float((video_quality or {}).get("face_detection_coverage", 0.0))
                ent = float((video_quality or {}).get("prediction_entropy", 99.0))
                if cov >= 0.8 and ent < 1.6:
                    lam_v = float(self.config.lambda_video)
                    p_fused = (1.0 - lam_v) * p_fused + lam_v * p_v
                    modality_weights["video"] = lam_v
                    modality_weights["text"] *= (1.0 - lam_v)
                    modality_weights["audio"] *= (1.0 - lam_v)
                    rule_applied += "+video_quality_injection"

        p_fused /= p_fused.sum()
        final_probs = {c: float(p_fused[i]) for i, c in enumerate(CLASS_NAMES)}
        final_pred_idx = int(p_fused.argmax())
        final_emotion = CLASS_NAMES[final_pred_idx]
        confidence = float(p_fused[final_pred_idx])

        return FusionResult(
            predicted_emotion=final_emotion,
            probabilities=final_probs,
            confidence=confidence,
            fusion_method=self.config.fusion_identifier,
            modalities_available=modalities_available,
            exclusion_reasons=exclusion_reasons,
            modality_weights=modality_weights,
            rule_applied=rule_applied,
            margin_audio=margin_audio,
        )

