"""Unit tests for Candidate 4 Text-Anchored Class-Conditional Bounded Fusion."""

from __future__ import annotations

import math
import numpy as np
import pytest

from api.fusion import (
    CLASS_NAMES,
    DEFAULT_KAPPA_AUDIO,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_TAU_AUDIO,
    FUSION_METHOD_IDENTIFIER,
    Candidate4FusionEngine,
    FusionConfig,
    _validate_probabilities,
)


def _uniform() -> list[float]:
    return [1.0 / 7.0] * 7


def _peaked(class_name: str, peak_val: float = 0.70) -> dict[str, float]:
    rest = (1.0 - peak_val) / 6.0
    return {c: (peak_val if c == class_name else rest) for c in CLASS_NAMES}


class TestProbabilityValidation:
    def test_valid_dict_mapping(self):
        d = _peaked("joy", 0.8)
        arr = _validate_probabilities(d)
        assert len(arr) == 7
        assert math.isclose(arr.sum(), 1.0, abs_tol=1e-5)
        assert arr[CLASS_NAMES.index("joy")] > 0.79

    def test_valid_numpy_array(self):
        arr_in = np.array([0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1])
        arr = _validate_probabilities(arr_in)
        assert math.isclose(arr.sum(), 1.0)

    def test_reject_missing_classes(self):
        with pytest.raises(ValueError, match="Missing classes"):
            _validate_probabilities({"anger": 1.0})

    def test_reject_wrong_dimension(self):
        with pytest.raises(ValueError, match="Expected 7 class probabilities"):
            _validate_probabilities([0.5, 0.5])

    def test_reject_negative_probabilities(self):
        with pytest.raises(ValueError, match="finite non-negative"):
            _validate_probabilities([-0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])

    def test_reject_unnormalized_probabilities(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            _validate_probabilities([0.1] * 7)  # sums to 0.7


class TestCandidate4FusionRules:
    @pytest.fixture
    def engine(self):
        return Candidate4FusionEngine()

    @pytest.mark.parametrize("protected_class", ["fear", "disgust", "sadness", "joy"])
    def test_rule1_protected_classes_lockout_auxiliary(self, engine, protected_class):
        """When text predicts a protected class, audio must be locked out completely."""
        text_probs = _peaked(protected_class, 0.65)
        # Even if audio is screaming anger with 99% confidence
        audio_probs = _peaked("anger", 0.95)

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.predicted_emotion == protected_class
        assert res.modality_weights["text"] == 1.0
        assert res.modality_weights["audio"] == 0.0
        assert res.rule_applied == f"protected_class_lockout_{protected_class}"
        # Probabilities match text exactly
        for c in CLASS_NAMES:
            assert math.isclose(res.probabilities[c], text_probs[c], abs_tol=1e-5)

    def test_rule2_acoustic_synergy_anger_injected(self, engine):
        """When text predicts neutral and audio exhibits high-confidence anger, audio is injected."""
        text_probs = _peaked("neutral", 0.50)
        # Anger peak 0.60, neutral is (1 - 0.60)/6 = 0.0667. Margin = 0.60 - 0.0667 = 0.5333 > 0.20
        audio_probs = _peaked("anger", 0.60)

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.rule_applied == "acoustic_synergy_anger"
        expected_lam = min(DEFAULT_LAMBDA_MAX, DEFAULT_KAPPA_AUDIO * res.margin_audio)
        assert math.isclose(res.modality_weights["audio"], expected_lam, abs_tol=1e-4)
        assert math.isclose(res.modality_weights["text"], 1.0 - expected_lam, abs_tol=1e-4)
        # Anger probability must have increased compared to text alone
        assert res.probabilities["anger"] > text_probs["anger"]

    def test_rule2_acoustic_synergy_surprise_injected(self, engine):
        """When text predicts neutral and audio exhibits surprise with margin > tau_audio, audio is injected."""
        text_probs = _peaked("neutral", 0.45)
        audio_probs = _peaked("surprise", 0.65)

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.rule_applied == "acoustic_synergy_surprise"
        assert res.modality_weights["audio"] > 0.0
        assert res.probabilities["surprise"] > text_probs["surprise"]

    def test_rule2_audio_below_margin_threshold_ignored(self, engine):
        """When audio predicts anger but margin over neutral <= tau_audio (0.20), audio is ignored."""
        text_probs = _peaked("neutral", 0.50)
        # Anger 0.25, neutral 0.20. Margin = 0.05 < 0.20
        audio_probs = {c: 0.11 for c in CLASS_NAMES}
        audio_probs["anger"] = 0.25
        audio_probs["neutral"] = 0.20
        audio_probs["surprise"] = 0.09
        tot = sum(audio_probs.values())
        audio_probs = {c: v / tot for c, v in audio_probs.items()}

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.rule_applied == "text_anchor_default"
        assert res.modality_weights["audio"] == 0.0
        assert res.modality_weights["text"] == 1.0
        assert res.predicted_emotion == "neutral"

    def test_rule2_non_synergy_audio_prediction_ignored(self, engine):
        """If audio predicts fear or joy, it must not inject audio into neutral text."""
        text_probs = _peaked("neutral", 0.55)
        audio_probs = _peaked("joy", 0.80)

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.rule_applied == "text_anchor_default"
        assert res.modality_weights["audio"] == 0.0
        assert res.predicted_emotion == "neutral"

    def test_missing_audio_handled_gracefully(self, engine):
        """When audio is omitted, fusion engine executes text-authoritative prediction."""
        text_probs = _peaked("neutral", 0.60)

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=None,
            audio_exclusion_reason="silent_audio (below -60 dB threshold)",
        )

        assert res.predicted_emotion == "neutral"
        assert res.modalities_available["audio"] is False
        assert res.modalities_available["text"] is True
        assert res.exclusion_reasons["audio"] == "silent_audio (below -60 dB threshold)"
        assert res.modality_weights["text"] == 1.0
        assert res.modality_weights["audio"] == 0.0

    def test_video_outside_default_production_path(self, engine):
        """In default configuration, lambda_video is 0.0 and video is excluded."""
        text_probs = _peaked("neutral", 0.60)
        video_probs = _peaked("joy", 0.90)

        res = engine.fuse(
            text_probabilities=text_probs,
            video_probabilities=video_probs,
        )

        assert res.modalities_available["video"] is False
        assert res.exclusion_reasons["video"] == "outside_production_path"
        assert res.modality_weights["video"] == 0.0

    def test_video_injection_when_explicitly_configured(self):
        """When lambda_video > 0 and quality thresholds are met, video contributes bounded weight."""
        config = FusionConfig(lambda_video=0.04)
        engine = Candidate4FusionEngine(config)
        text_probs = _peaked("neutral", 0.50)
        video_probs = _peaked("joy", 0.70)

        res = engine.fuse(
            text_probabilities=text_probs,
            video_probabilities=video_probs,
            video_quality={"face_detection_coverage": 0.90, "prediction_entropy": 1.2},
        )

        assert res.modalities_available["video"] is True
        assert res.modality_weights["video"] == 0.04
        assert math.isclose(res.modality_weights["text"], 0.96)
        assert "video_quality_injection" in res.rule_applied

    def test_fusion_metadata_attributes(self, engine):
        text_probs = _peaked("anger", 0.60)
        res = engine.fuse(text_probs)
        assert res.fusion_method == FUSION_METHOD_IDENTIFIER
        assert set(res.probabilities.keys()) == set(CLASS_NAMES)
        assert math.isclose(sum(res.probabilities.values()), 1.0)
        assert 0.0 <= res.confidence <= 1.0

    def test_margin_definition_neutral_contrast_vs_top2_divergence(self, engine):
        """Regression test verifying Candidate 4 uses M_a = p_a[pred] - p_a[neutral].
        
        Under M_a = p_a[pred] - p_a[neutral]:
          M_a = 0.38 - 0.10 = 0.28 > 0.20 -> TRIGGERS acoustic synergy.
        Under M_a = top1 - top2:
          M_a = 0.38 - 0.28 = 0.10 <= 0.20 -> WOULD NOT trigger synergy.
        Candidate 4 requires the neutral contrast formula from Phase 13.
        """
        text_probs = _peaked("neutral", 0.45)
        # Audio distribution where anger=0.38, joy=0.28, neutral=0.10, rest=0.06
        audio_probs = {
            "anger": 0.38,
            "disgust": 0.06,
            "fear": 0.06,
            "joy": 0.28,
            "neutral": 0.10,
            "sadness": 0.06,
            "surprise": 0.06,
        }

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.rule_applied == "acoustic_synergy_anger"
        assert res.margin_audio == pytest.approx(0.28, abs=1e-4)
        # Expected lambda_a = min(0.40, 1.5 * 0.28) = min(0.40, 0.42) = 0.40
        assert res.modality_weights["audio"] == pytest.approx(0.40, abs=1e-4)
        assert res.modality_weights["text"] == pytest.approx(0.60, abs=1e-4)

    def test_margin_definition_real_dev_sample_divergence(self, engine):
        """Regression test using exact probabilities from dev sample dev_dia104_utt10.
        
        top1=surprise (0.305), 2nd=joy (0.231), neutral=0.101.
        M_neutral = 0.305 - 0.101 = 0.204 > 0.20 (synergy activated).
        M_top2    = 0.305 - 0.231 = 0.074 <= 0.20 (synergy inactive).
        """
        text_probs = _peaked("neutral", 0.50)
        audio_probs = {
            "anger": 0.091,
            "disgust": 0.052,
            "fear": 0.063,
            "joy": 0.231,
            "neutral": 0.101,
            "sadness": 0.157,
            "surprise": 0.305,
        }

        res = engine.fuse(
            text_probabilities=text_probs,
            audio_probabilities=audio_probs,
        )

        assert res.rule_applied == "acoustic_synergy_surprise"
        assert res.margin_audio == pytest.approx(0.204, abs=1e-3)
        expected_lam = min(0.40, 1.50 * 0.204)
        assert res.modality_weights["audio"] == pytest.approx(expected_lam, abs=1e-3)


