"""Phase 18D multimodal extension preserving Candidate 4 Text+Speech unchanged."""

from __future__ import annotations

from typing import Any

import numpy as np

from api.fusion import (
    CLASS_NAMES,
    Candidate4FusionEngine,
    FusionConfig,
    FusionResult,
    _validate_probabilities,
)

EXTENDED_FUSION_IDENTIFIER = "candidate4_plus_validated_resnet18_video_v1"
VIDEO_ONLY_IDENTIFIER = "single_modality_video_resnet18_v1"
AUDIO_ONLY_IDENTIFIER = "single_modality_audio_wav2vec2_v1"
AUDIO_VIDEO_FUSION_IDENTIFIER = "audio_primary_plus_validated_resnet18_video_v1"

VALIDATED_LAMBDA_VIDEO = 0.08
MIN_VIDEO_COVERAGE = 0.40
MAX_VIDEO_ENTROPY = 1.85


class ExtendedMultimodalFusionEngine:
    """Separately validated Phase 18D extension layered cleanly over Candidate 4.

    Guarantees:
    1. Text-only and Text+Speech requests without video delegate verbatim to
       Candidate4FusionEngine (0% behavior or weight drift).
    2. When a valid ResNet18 facial prediction is supplied alongside Text(+Audio) and
       passes dev-validated quality thresholds (coverage >= 0.40, entropy <= 1.85),
       applies bounded weight w_v = 0.08 (improving MELD dev NLL, Brier score, and ECE
       with zero Accuracy or Macro/Weighted F1 degradation).
    3. If video fails quality gates, it remains reported in individual predictions and
       advisory telemetry with w_v = 0.0 (no harmful injection).
    4. Supports Video-only, Audio-only, and Audio+Video requests when Text is omitted,
       re-normalizing weights strictly across available non-fabricated modalities.
    """

    def __init__(
        self,
        base_engine: Candidate4FusionEngine | None = None,
        lambda_video: float = VALIDATED_LAMBDA_VIDEO,
        min_video_coverage: float = MIN_VIDEO_COVERAGE,
        max_video_entropy: float = MAX_VIDEO_ENTROPY,
    ) -> None:
        self.base_engine = base_engine or Candidate4FusionEngine(FusionConfig(lambda_video=0.0))
        self.lambda_video = float(lambda_video)
        self.min_video_coverage = float(min_video_coverage)
        self.max_video_entropy = float(max_video_entropy)

    def _video_passes_quality(self, v_probs: np.ndarray, video_quality: dict[str, Any] | None) -> tuple[bool, str | None]:
        if not video_quality:
            return True, None
        cov = video_quality.get("face_detection_coverage")
        if cov is not None and float(cov) < self.min_video_coverage:
            return False, f"video_low_face_coverage ({float(cov):.2f} < {self.min_video_coverage:.2f})"
        ent = video_quality.get("prediction_entropy")
        if ent is None:
            ent = float(-(v_probs * np.log(np.maximum(v_probs, 1e-12))).sum())
        if float(ent) > self.max_video_entropy:
            return False, f"video_high_entropy ({float(ent):.2f} > {self.max_video_entropy:.2f})"
        return True, None

    @staticmethod
    def _finalize(
        probs: np.ndarray,
        fusion_method: str,
        modalities_available: dict[str, bool],
        exclusion_reasons: dict[str, str | None],
        modality_weights: dict[str, float],
        rule_applied: str,
        margin_audio: float | None = None,
    ) -> FusionResult:
        probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
        total = float(probs.sum())
        if not np.isfinite(probs).all() or total <= 0.0:
            raise ValueError("Fused probability vector must be finite with positive sum.")
        probs /= total
        pred_idx = int(np.argmax(probs))
        return FusionResult(
            predicted_emotion=CLASS_NAMES[pred_idx],
            probabilities={cls_name: float(probs[i]) for i, cls_name in enumerate(CLASS_NAMES)},
            confidence=float(probs[pred_idx]),
            fusion_method=fusion_method,
            modalities_available=modalities_available,
            exclusion_reasons=exclusion_reasons,
            modality_weights=modality_weights,
            rule_applied=rule_applied,
            margin_audio=margin_audio,
        )

    def fuse(
        self,
        text_probabilities: dict[str, float] | np.ndarray | None = None,
        audio_probabilities: dict[str, float] | np.ndarray | None = None,
        video_probabilities: dict[str, float] | np.ndarray | None = None,
        audio_exclusion_reason: str | None = None,
        video_exclusion_reason: str | None = None,
        video_quality: dict[str, Any] | None = None,
        text_exclusion_reason: str | None = None,
    ) -> FusionResult:
        # Path 1: Authoritative Text provided (Candidate 4 preserved verbatim when video absent)
        if text_probabilities is not None:
            base_res = self.base_engine.fuse(
                text_probabilities=text_probabilities,
                audio_probabilities=audio_probabilities,
                video_probabilities=None,
                audio_exclusion_reason=audio_exclusion_reason,
                video_exclusion_reason=video_exclusion_reason,
                video_quality=None,
            )

            if video_probabilities is None:
                reasons = dict(base_res.exclusion_reasons)
                reasons["video"] = video_exclusion_reason or "No video file provided"
                return FusionResult(
                    predicted_emotion=base_res.predicted_emotion,
                    probabilities=base_res.probabilities,
                    confidence=base_res.confidence,
                    fusion_method=base_res.fusion_method,
                    modalities_available=base_res.modalities_available,
                    exclusion_reasons=reasons,
                    modality_weights=base_res.modality_weights,
                    rule_applied=base_res.rule_applied,
                    margin_audio=base_res.margin_audio,
                )

            v_arr = _validate_probabilities(video_probabilities)
            passed, fail_reason = self._video_passes_quality(v_arr, video_quality)

            if not passed or self.lambda_video <= 0.0:
                reasons = dict(base_res.exclusion_reasons)
                reasons["video"] = fail_reason
                avail = dict(base_res.modalities_available)
                avail["video"] = True
                return FusionResult(
                    predicted_emotion=base_res.predicted_emotion,
                    probabilities=base_res.probabilities,
                    confidence=base_res.confidence,
                    fusion_method=EXTENDED_FUSION_IDENTIFIER,
                    modalities_available=avail,
                    exclusion_reasons=reasons,
                    modality_weights=dict(base_res.modality_weights),
                    rule_applied=f"{base_res.rule_applied}+video_advisory_only",
                    margin_audio=base_res.margin_audio,
                )

            c4_arr = np.array([base_res.probabilities[c] for c in CLASS_NAMES], dtype=np.float64)
            w_v = float(self.lambda_video)
            scale = 1.0 - w_v
            fused = scale * c4_arr + w_v * v_arr

            reasons = dict(base_res.exclusion_reasons)
            reasons["video"] = None
            avail = dict(base_res.modalities_available)
            avail["video"] = True
            weights = {
                "text": float(base_res.modality_weights["text"] * scale),
                "audio": float(base_res.modality_weights["audio"] * scale),
                "video": w_v,
            }
            return self._finalize(
                probs=fused,
                fusion_method=EXTENDED_FUSION_IDENTIFIER,
                modalities_available=avail,
                exclusion_reasons=reasons,
                modality_weights=weights,
                rule_applied=f"{base_res.rule_applied}+validated_video_calibrated",
                margin_audio=base_res.margin_audio,
            )

        # Path 2: No-Text requests (Audio only, Video only, or Audio + Video)
        t_reason = text_exclusion_reason or "No text input provided"
        if audio_probabilities is None and video_probabilities is None:
            raise ValueError("At least one valid modality (text, audio, or video) must be provided.")

        if audio_probabilities is not None and video_probabilities is None:
            a_arr = _validate_probabilities(audio_probabilities)
            return self._finalize(
                probs=a_arr,
                fusion_method=AUDIO_ONLY_IDENTIFIER,
                modalities_available={"text": False, "audio": True, "video": False},
                exclusion_reasons={
                    "text": t_reason,
                    "audio": None,
                    "video": video_exclusion_reason or "No video file provided",
                },
                modality_weights={"text": 0.0, "audio": 1.0, "video": 0.0},
                rule_applied="audio_only_fallback",
            )

        if audio_probabilities is None and video_probabilities is not None:
            v_arr = _validate_probabilities(video_probabilities)
            return self._finalize(
                probs=v_arr,
                fusion_method=VIDEO_ONLY_IDENTIFIER,
                modalities_available={"text": False, "audio": False, "video": True},
                exclusion_reasons={
                    "text": t_reason,
                    "audio": audio_exclusion_reason or "No audio file provided",
                    "video": None,
                },
                modality_weights={"text": 0.0, "audio": 0.0, "video": 1.0},
                rule_applied="video_only_resnet18",
            )

        # Both Audio and Video without Text
        a_arr = _validate_probabilities(audio_probabilities)
        v_arr = _validate_probabilities(video_probabilities)
        passed, fail_reason = self._video_passes_quality(v_arr, video_quality)
        w_v = float(self.lambda_video) if (passed and self.lambda_video > 0.0) else 0.0
        w_a = 1.0 - w_v
        fused = w_a * a_arr + w_v * v_arr
        rule = "audio_video_calibrated_blend" if w_v > 0.0 else "audio_primary_video_advisory_only"
        return self._finalize(
            probs=fused,
            fusion_method=AUDIO_VIDEO_FUSION_IDENTIFIER,
            modalities_available={"text": False, "audio": True, "video": True},
            exclusion_reasons={"text": t_reason, "audio": None, "video": fail_reason},
            modality_weights={"text": 0.0, "audio": w_a, "video": w_v},
            rule_applied=rule,
        )

